#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Copyright 2019 Julian Schwanbeck (julian.schwanbeck@med.uni-goettingen.de)
https://github.com/schwanbeck/YSMR
##Explanation
This file contains the main functions used by YSMR.
This file is part of YSMR. YSMR is free software: you can distribute it and/or modify
it under the terms of the GNU General Public License as published by the Free Software Foundation,
either version 3 of the License, or (at your option) any later version. YSMR is distributed in
the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
details. You should have received a copy of the GNU General Public License along with YSMR. If
not, see <http://www.gnu.org/licenses/>.
"""
import logging
import multiprocessing as mp
import os
import sys
from datetime import datetime
from time import strftime, localtime, strptime

import cv2
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.signal import medfilt
from scipy.spatial import distance as dist

from helper_file import (
    _backup,
    _mkdir,
    different_tracks,
    elapsed_time,
    find_paths,
    # get_colour_map,
    get_configs,
    get_data,
    get_loggers,
    reshape_result,
    save_list,
    sort_list,
)
from tracker import CentroidTracker


def track_bacteria(curr_path, settings=None):
    logger = logging.getLogger('ei').getChild(__name__)
    '''
    Used settings:
    settings['minimal frame count']
    settings['verbose']
    settings['frames per second']
    settings['store processed .csv file']
    settings['debugging']
    settings['display video analysis']
    settings['save video']
    settings['stop evaluation on error']
    settings['color filter']
    settings['white bacteria on dark background']
    settings['threshold offset for detection']
    settings['include luminosity in tracking calculation']
    settings['list save length interval']
    settings['delete .csv file after analysis']
    settings['evaluate files after analysis']
    '''
    t_one_track_bacteria = datetime.now()
    if settings is None:
        settings = get_configs()
    if settings is None:
        logger.critical('No settings provided / could not get settings for track_bacteria().')
        return None
    # We may have to set the log level/loggers again due to multiprocessing
    get_loggers(
        log_level=settings['log_level'],
        logfile_name=settings['log file path'],
        short_stream_output=settings['shorten displayed logging output'],
        short_file_output=settings['shorten logfile logging output'],
        log_to_file=settings['log to file'])
    # Log some general stuff
    logger.debug('Starting process - module: {} PID: {}'.format(__name__, os.getpid()))
    # Check for errors
    if not os.path.isfile(curr_path):
        logger.critical('File {} does not exist'.format(curr_path))
        return None
    try:
        cap = cv2.VideoCapture(curr_path)
    except (IOError, OSError) as io_error:
        logger.exception('Cannot open file {} due to error: {}'.format(curr_path, io_error))
        return None

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count < settings['minimal frame count']:
        logger.warning('File {} too short; file was skipped.'.format(curr_path))
        return None
    try:
        fps_of_file = cap.get(cv2.CAP_PROP_FPS)
        if settings['verbose'] or fps_of_file != settings['frames per second']:
            logger.info('fps of file: {}'.format(fps_of_file))
    except Exception as ex:
        template = 'An exception of type {0} occurred while accessing fps from file {2}. Arguments:\n{1!r}'
        logger.exception(template.format(type(ex).__name__, ex.args, curr_path))
        if settings['frames per second'] <= 0:
            logger.critical('User defined fps unacceptable: type: {} value: {}'.format(
                type(settings['frames per second']), settings['frames per second']))
            return None
        else:
            fps_of_file = settings['frames per second']
    finally:
        pass
    test = settings['debugging']

    pathname, filename = os.path.split(curr_path)
    list_name = save_list(get_name=True, file_path=pathname, filename=filename)
    logger.info('Starting with file {}'.format(curr_path))

    # Set initial values; initialise result list
    old_list = save_list(file_path=pathname, filename=filename,
                         first_call=True, store_old_list=settings['store processed .csv file'])
    # Save old_list_name for later if it exists; False otherwise
    ct = CentroidTracker()  # Initialise tracker instance
    coords = []  # Empty list to store calculated coordinates
    curr_frame_count = 0
    # skip_frames = 0
    total_threshold = 0  # Detection threshold total at which gray value bacteria are detected;
    # calculated differently depending on white_bac_on_black_bgr
    fps_total = []  # List of calculated fps
    curr_threshold = 0  # Current threshold (calculated on the first 600 frames; thereafter left as is)
    error_during_read = False  # Set to true if some errors occur; used to restore old list afterwards if it exists
    (objects, degrees) = (None, None)  # reset objects, additional_info (probably useless but doesn't hurt)

    if test and settings['display video analysis']:  # Display first frame in case frame-by-frame analysis is necessary
        ret, frame = cap.read()
        cv2.imshow('{}'.format(filename), frame)
    (frame_height, frame_width) = (int(cap.get(4)),
                                   int(cap.get(3)))  # Image dimensions
    if settings['verbose']:
        logger.debug('Frame height: {}, width: {}'.format(frame_height, frame_width))

    # Background removal:
    # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    if settings['save video']:
        output_video_name = '{}/{}_output.avi'.format(pathname, filename)
        logger.info('Output video file: {}'.format(output_video_name))
        out = cv2.VideoWriter(output_video_name,
                              cv2.VideoWriter_fourcc(*'MJPG'),  # Codec *'MJPG'  # X264
                              fps_of_file,  # FPS
                              (frame_width, frame_height)  # Dimensions
                              )
    # min_frame_count += skip_frames
    while True:  # Loop over video
        # if cv2.waitKey(1) & 0xFF == ord('n'):  # frame-by-frame (Put entire following block into this if clause)
        timer = cv2.getTickCount()
        ret, frame = cap.read()  # ret: True/False, depends on whether another frame could be retrieved
        # frame: the actual current frame

        # @todo: skip frames?
        # if curr_frame_count < skip_frames:
        #     continue  # skip frame/jump back to start
        # uMatframe = cv2.UMat(frame)

        # Stop conditions
        if not ret and (frame_count == curr_frame_count + 1 or  # some file formats skip one frame
                        frame_count == curr_frame_count) and frame_count >= settings['minimal frame count']:
            # If a frame could not be retrieved and the minimum frame nr. has been reached
            logger.info('Frames from file {} read.'.format(filename))
            break
        elif not ret:  # Something must've happened, user decides if to proceed
            logger.critical('Error during cap.read() with file {}'.format(curr_path))
            error_during_read = settings['stop evaluation on error']
            break

        '''
        SORT THIS SHIT OUT  >  shit sorted, mebbeh?
        gray -> Threshold direct, threshold fixed at 83 / 68?
        gray -> clahe -> Threshold --> can always use same threshold?
        mean, stddev = cv2.meanStdDev(src[, mean[, stddev[, mask]]]) -> find threshold dynamically works
        gray -> blurred -> Threshold (standard?)
        '''
        # frame = imutils.resize(frame, width=600)  # Loss of information; harder to detect stuff, gain of speed?
        # Resizing not necessary and gain in speed doesn't outweigh loss of precision

        gray = cv2.cvtColor(frame, settings['color filter'])  # Convert to gray scale

        if curr_frame_count <= settings['minimal frame count']:  # set threshold adaptively  + skip_frames
            mean, stddev = cv2.meanStdDev(gray)
            if settings['white bacteria on dark background']:
                total_threshold += (mean + stddev + settings['threshold offset for detection'])
                # Bacteria are brighter than background
            else:
                total_threshold += (mean - stddev - settings['threshold offset for detection'])
                # Bacteria are darker than background
                # It's sadly not simply (255 - threshold)
            curr_threshold = int(total_threshold / (curr_frame_count + 1))  # average input  - skip_frames
            # if your threshold is not ok after 600 frames either install
            # a new power source/lamp in your microscope or
            # stop fiddling with brightness during recording
            # Usually stable after ~2 frames
            if curr_frame_count == settings['minimal frame count']:
                logger.debug('Background threshold level: {} (of 255), '
                             'mean: {:.2f}, std. deviation: {:.2f}, offset: {}'.format(
                              curr_threshold, mean.item(), stddev.item(), settings['threshold offset for detection']))
        # various other tries to optimise threshold:
        # blurred = cv2.bilateralFilter(gray, 3, 75, 75)
        # equ = clahe.apply(gray)  # uncomment clahe above; background removal
        # blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        # blurred = cv2.medianBlur(gray, 5)

        # UMat: should utilise graphics card; tends to slow down the whole thing a lot
        # Presumably because a) my graphics card is rubbish, b) the conversion takes too much time,
        # c) subsequent calculations aren't exactly witchcraft, d) all of the above
        # gray = cv2.UMat(gray)

        blurred = cv2.GaussianBlur(gray, (3, 3), 0)

        if settings['white bacteria on dark background']:
            # All pixels above curr_threshold are set to 255 (white); others are set to 0
            thresh = cv2.threshold(blurred, curr_threshold, 255, cv2.THRESH_BINARY)[1]
        else:
            thresh = cv2.threshold(blurred, curr_threshold, 255, cv2.THRESH_BINARY_INV)[1]
            # Simply inverse output (as above)

        # Other threshold variations; proved unnecessary:
        # thresh = cv2.threshold(blurred, 90, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
        # thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 1)

        # display individual conversion steps to see which one fucks up
        if settings['debugging'] and settings['display video analysis']:
            # cv2.imshow('frame', frame)
            # cv2.imshow('gray', gray)
            # cv2.imshow('equ', equ)
            cv2.imshow('blurred', blurred)
            cv2.imshow('threshold', thresh)

        contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # returns image, contours, hierarchy (cv2 v.3.4.5.20); we just care about contours
        # or returns contours, hierarchy (cv2 v4+)
        if len(contours) == 3:
            contours = contours[1]
        elif len(contours) == 2:
            contours = contours[0]
        else:
            logger.critical('Unexpected return value from cv2.findContours(); tuple must have length of 2 or 3. '
                            'Check openCV documentation for cv2.findContours().')
            return None

        rects = []  # List of bounding rectangles in the current thresholded frame
        for contour in contours:
            reichtangle = cv2.minAreaRect(contour)
            if settings['include luminosity in tracking calculation']:
                box = np.int0(cv2.boxPoints(reichtangle))
                # must be generated for each object as cv2.fillpoly() changes it's input
                mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
                cv2.fillPoly(mask, [box], 255)
                # Average intensity; between 0 and 255; at around 100 on average
                reichtangle_mean = (cv2.mean(gray, mask)[0]) / 100
                # Divided by 100 to give a value from 0.00 to 2.55: thus it only weighs heavy in our distance matrix
                # at distances < 2.56 px; thereby hopefully helping miss-assignment after overlap
                # at least that's the intention
                rects.append(reshape_result(reichtangle, reichtangle_mean))
                # tracker.py has been remodelled to adaptively take n dimensions for distance matrix
            else:
                rects.append(reshape_result(reichtangle))
                # reshape_result(tuple_of_tuples) returns ((x, y[, *args]), (w, h, degrees_orientation))

            if settings['display video analysis'] or settings['save video']:  # Display bounding boxes
                box = np.int0(cv2.boxPoints(reichtangle))
                cv2.drawContours(frame, [box], -1, (255, 0, 0), 0)

        objects, wh_degrees = ct.update(rects)  # Calls CentroidTracker.update() from tracker.py

        for index, (objectID, centroid) in enumerate(objects.items()):  # object.items() loop
            # Follow the KISS principle (fancy smoother option is surely available, but this works):
            # Append results to list (Frame, ID, x, y, (other values))
            coords.append((curr_frame_count, objectID, centroid, wh_degrees[objectID]))

            # draw both the ID of the object and the center point
            if settings['display video analysis'] or settings['save video']:  # and objectID == curr_bac:
                text = '{}'.format(objectID)
                # Display object ID:
                cv2.putText(frame, text, (int(centroid[0]) - 10, int(centroid[1]) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 0)
                # Display centroid:
                cv2.circle(frame, (int(centroid[0]), int(centroid[1])), 0, (0, 255, 0), -1)

        # Frame is finished, video can be saved
        if settings['save video']:
            cv2.putText(frame,  # image
                        '{}'.format(filename[:].replace('_', ' ')),  # text
                        (20, 20),  # xy coordinates
                        cv2.FONT_HERSHEY_SIMPLEX,  # font
                        0.7,  # text size
                        (220, 220, 60),  # colour
                        1  # line thickness
                        )
            out.write(frame)

        # Change coords.list if it is long enough (I/O-operations are time consuming)
        if len(coords) >= settings['list save length interval']:
            # shift this block into previous for-loop if too many objects per frame causes problems
            # change list_save_length in tracking.ini if current value is an issue.
            # send coords off to be saved on drive:
            save_list(coords=coords, file_path=pathname, filename=filename)
            coords = []  # reset coords list

        curr_frame_count += 1  # increase frame counter
        # Calculate curr. fps - jumps around a lot due to intermittent save_list()
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
        fps_total.append(fps)  # for average fps later

        if settings['display video analysis']:  # Display current FPS on frame
            cv2.putText(frame,  # image
                        'FPS: {}'.format(int(fps)),  # text
                        (100, 50),  # xy coordinates
                        cv2.FONT_HERSHEY_SIMPLEX,  # font
                        0.75,  # text size
                        (50, 50, 170),  # colour
                        2  # line thickness
                        )
            cv2.imshow('{}'.format(filename), frame)  # Display the image
            if cv2.waitKey(1) & 0xFF == ord('q'):  # Interrupt display on 'q'-keypress
                error_during_read = True
                logger.error('Processing file interrupted by user: {}'.format(curr_path))
                break
        # @todo: break_time_seconds from .ini - for partial analysis?
        # break_time_seconds = 0
        # if break_time_seconds != 0 and curr_frame_count > (break_time_seconds * fps_of_file):
        #     logger.debug('Video analysis cancelled early at frame {} ({} sec)'.format(
        #         curr_frame_count, (break_time_seconds * fps_of_file)))
        #     break

    if coords:  # check if list is not empty ([] == False, otherwise True)
        save_list(coords=coords, file_path=pathname, filename=filename)  # Save the remainder

    if settings['save video']:
        out.release()
    cap.release()
    cv2.destroyAllWindows()  # Close active windows

    if old_list and error_during_read:  # Strings evaluate to true, otherwise false; restores old results
        try:
            os.remove(list_name)
            os.rename(old_list, list_name)
            logger.info('Restoring old list: {}'.format(list_name))
        except (OSError, FileNotFoundError) as file_removal_error:
            logger.error('An exception of type {0} occurred with path {1}. Arguments:\n{2!r}'.format(
                type(file_removal_error).__name__, list_name, file_removal_error.args))
        finally:
            pass

    last_object_id = next(reversed(objects))  # get number of last object
    df_for_eval = sort_list(file_path=list_name, save_file=not settings['delete .csv file after analysis'])

    logger.info('fps: {}, objects: {}, frames: {}, csv: {}'.format(  # Display some infos
        '{:.2f}'.format((sum(fps_total) / curr_frame_count)).rjust(6, ' '),  # Average FPS
        '{}'.format(last_object_id + 1).rjust(6, ' '),  # Total Nr. of objects
        '{:>6} of {:>6}'.format(curr_frame_count, frame_count),  # Current frames / total frames
        list_name  # results
    ))

    if error_during_read:
        logger.critical('Error during read, stopping before evaluation. File: {}'.format(curr_path))
        return None
    else:
        if settings['evaluate files after analysis']:
            logger.info('Starting evaluation of file {}'.format(list_name))
            start_it_up(path_to_files=list_name,
                        df=df_for_eval,
                        fps=fps_of_file,
                        frame_height=frame_height,
                        frame_width=frame_width,
                        settings=settings,
                        create_logger=False,
                        )
        if settings['delete .csv file after analysis']:
            logger.info('Removing .csv file: {}'.format(list_name))
            try:
                os.remove(list_name)
            except (OSError, FileNotFoundError):
                logger.warning('Could not delete file {}'.format(list_name))
            finally:
                pass
        # If everything went well, hand over the list name to the next process
        logger.info('Finished process - module: {} PID: {}, elapsed time: {}'.format(
            __name__,
            os.getpid(),
            elapsed_time(t_one_track_bacteria))
        )
    return list_name


def find_good_tracks(df_passed, lower_boundary, upper_boundary, start, stop,
                     frame_height, frame_width, settings, recursion=0):
    logger = logging.getLogger('ei').getChild(__name__)
    size = stop - start + 1
    # avoid off-by-one error; used instead of df.shape[0] to avoid impossible calls to df.iloc[:]
    '''
    # kick_reason:
    7: size < 600 
    6: holes > 6  (not used, as larger track halve will be re-analysed)
    5: distance outlier
    4: duration*1.05 > size
    3: average area not within bounds
    2: average ratio not within bounds
    1: average x/y not within bounds
    0: pass
    
    Used settings:
    settings['minimal length in seconds']
    settings['maximal consecutive holes']
    settings['maximal empty frames in %']
    settings['average width/height ratio min.']
    settings['average width/height ratio max.']
    settings['percent of screen edges to exclude']
    settings['maximal recursion depth (0 is off)']
    '''
    kick_reason = 7
    return_result = []
    sub_part = []
    # Too short tracks aren't useful and can immediately be discarded
    if size >= settings['minimal length in seconds']:  # 20s @30 fps  # :(2.5 * 30)
        # Do not allow tracking holes for more than 4 frames, try to find useful halves otherwise
        kick_reason -= 1
        # We keep df_passed for later
        df = df_passed.iloc[start:stop + 1]  # iloc upper range in slice is exclusive
        size = df.shape[0]  # reassigned just in case
        look_for_holes = df['POSITION_T'].diff()
        if look_for_holes.max() <= settings['maximal consecutive holes']:
            kick_reason -= 1
            # Check if there are no outliers (at pos. 0 would be ok)
            if df['distance'].sum() == 0:
                kick_reason -= 1
                # remove tracks with too big holes (>=5%)
                # calculate difference between first and last frame
                duration = df['POSITION_T'].iloc[-1] - df['POSITION_T'].iloc[0] + 1  # avoid off-by-one error
                duration_size_ratio = duration / size
                # logger.debug(duration_size_ratio)
                if duration_size_ratio < settings['maximal empty frames in %']:
                    kick_reason -= 1
                    # Average area should be within 0.1/0.9 quartile
                    # off all detections if mostly bacteria are detected
                    if lower_boundary <= df['area'].mean() <= upper_boundary:
                        kick_reason -= 1
                        # Rod-shaped bacteria should have a w/h ratio of 1:8 to roughly 1:1.5
                        if settings['average width/height ratio min.'] < df['ratio_wh'].mean() < \
                                settings['average width/height ratio max.']:
                            # 0.125 < average_ratio < 0.67:
                            kick_reason -= 1
                            # exclude 5% of screen edges
                            if settings['percent of screen edges to exclude'] * frame_height \
                                    < df['POSITION_Y'].mean() < (
                                    1 - settings['percent of screen edges to exclude']) * frame_height:
                                if settings['percent of screen edges to exclude'] * frame_width \
                                        < df['POSITION_X'].mean() < (
                                        1 - settings['percent of screen edges to exclude']) * frame_width:
                                    kick_reason -= 1
                                    # everything is as it should be, append start/stop to return_result
                                    return_result.append((start, stop))
            # Halve track into part before/after outlier; try to analyze those instead; extend return_result
            else:
                # Hole is index where distance_outlier == 1; hole is excluded
                idx_outlier = df['distance'].idxmax()
                sub_part.extend([(start, idx_outlier - 1), (idx_outlier + 1, stop)])
        # Halve track into part before/after hole; try to analyze those instead; extend return_result
        else:
            # Hole is index of largest number in df['POSITION_T'].diff()
            hole_idx = look_for_holes.idxmax()
            sub_part.extend([(start, hole_idx - 1), (hole_idx, stop)])
    elif recursion >= settings['maximal recursion depth (0 is off)'] != 0:
        logger.debug('Recursion reached max. level at TRACK_ID: {} start: {} stop: {}'.format(
            df_passed.loc[start, 'TRACK_ID'], start, stop))
    # stop recursions from causing stack overflows
    if sub_part and recursion < settings['maximal recursion depth (0 is off)']:
        kick_reason_list = [kick_reason]
        for (sub_start, sub_stop) in sub_part:
            if sub_stop - sub_start + 1 < settings['minimal length in seconds']:
                continue  # Skip to stop unnecessary recursions
            sub_return_result, kick_reason = find_good_tracks(
                df_passed=df_passed,
                lower_boundary=lower_boundary,
                upper_boundary=upper_boundary,
                settings=settings,
                start=sub_start,
                stop=sub_stop,
                frame_height=frame_height,
                frame_width=frame_width,
                recursion=recursion + 1,
            )
            kick_reason_list.append(kick_reason)
            return_result.extend(sub_return_result)
        # Get the smallest kick reason (default kick_reason, but sub-part might've gotten further)
        kick_reason = min(kick_reason_list)
    return return_result, kick_reason


def single_plots_in_your_area(df, plot_name, px_to_micrometre, offset=0, period=1, fps=30):
    logger = logging.getLogger('ei').getChild(__name__)
    logger.debug('Creating single track displays')
    # Base values
    # moving_average = 10
    # avg_mask = np.ones(moving_average) / moving_average
    # colors = get_colour_map(counts=6)
    # if list_of_selected_plots is None:
    df.reset_index(inplace=True)
    diff_tracks_start, diff_tracks = different_tracks(df)
    list_of_selected_plots = []
    for idx in range(0, len(diff_tracks)):
        if idx == 0:
            list_of_selected_plots.append((0, diff_tracks[idx]))
        else:
            list_of_selected_plots.append((diff_tracks[idx - 1] + 1, diff_tracks[idx]))
    df['travelled_dist'] = np.sqrt(np.square(df['x_delta']) + np.square(df['y_delta'])) / px_to_micrometre
    # df.loc[df['travelled_dist'].isnull(), ['travelled_dist']] = 0
    # np.put(df['travelled_dist'], diff_tracks_start, 0)
    df['travelled_dist'] = df['travelled_dist'] / df['t_delta']
    # Initialise plots
    fig, plots = plt.subplots(len(list_of_selected_plots), 2, gridspec_kw={'width_ratios': [5, 1]})  #
    fig.set_size_inches(11.6929133858, 11.6929133858)  # (8.2677165354, 11.6929133858)

    # angle_diff = 10
    min_angle = 30

    df['x_stop'] = np.where(df['minimum'] == 0, df['POSITION_X'], np.NaN)
    df['y_stop'] = np.where(df['minimum'] == 0, df['POSITION_Y'], np.NaN)

    df['x_change'] = np.where((df['angle_calc'] > min_angle) & (df['minimum'] == 1),
                              df['POSITION_X'], np.NaN)
    df['y_change'] = np.where((df['angle_calc'] > min_angle) & (df['minimum'] == 1),
                              df['POSITION_Y'], np.NaN)

    # Go through selected plots
    for idx, (start, stop) in enumerate(list_of_selected_plots):
        # Select df
        df_curr = df.loc[start:stop].copy()
        # Cumulative distance

        df_curr['cum_sum'] = df_curr.loc[:, ['travelled_dist']].cumsum()
        # Velocity
        df_curr['velocity'] = df_curr['travelled_dist']
        df_curr['velocity'] = np.where(df_curr['travelled_dist'] > 10 ** -3, df_curr['travelled_dist'], 0)
        # df_curr.loc[start, ['velocity']] = 0
        # df_curr.loc[df_curr['velocity'].isnull(), ['velocity']] = 0
        for kernel_size in [3, 31]:  # range(0, 3):
            df_curr['velocity'] = medfilt(df_curr['velocity'], kernel_size=kernel_size)

        df_curr.loc[start, ['velocity']] = 0
        change_array = df_curr['velocity'].values
        # start / stop
        change_to_move = np.where(
            (change_array[:-1] != change_array[1:]) &
            (change_array[:-1] == 0)
        )[0]

        change_to_stop = np.where(
            (change_array[:-1] != change_array[1:]) &
            (change_array[1:] == 0)
        )[0]
        # Acceleration
        # df_curr['acceleration'] = df_curr.loc[:, ['velocity']].diff()
        # df_curr.loc[df_curr['acceleration'].isnull(), ['acceleration']] = 0
        # df_curr['acceleration'] = np.convolve(df_curr['acceleration'], avg_mask, 'same')
        # df_curr['acceleration'] = np.convolve(df_curr['acceleration'], avg_mask, 'same')
        # df_curr.loc[start, ['acceleration']] = 0

        t = [t - df_curr.loc[start, 'POSITION_T'] for t in df_curr['POSITION_T'].array]  # Set t0 to 0
        x = df_curr['POSITION_X']
        y = df_curr['POSITION_Y']
        x_change = df_curr['x_change']
        y_change = df_curr['y_change']
        x_stop = df_curr['x_stop']
        y_stop = df_curr['y_stop']

        # d = df_curr['cum_sum']
        v = df_curr['velocity']
        # a = df_curr['acceleration']

        angle = df_curr['angle_calc']

        ax, bx = plots[idx]
        color = 'tab:blue'
        if idx == len(list_of_selected_plots) - 1:
            ax.set_xlabel('time ({:.2f} s)'.format(1 / fps * period))
        ax.set_ylabel('velocity', color=color)
        ax.plot(t[:], v, color=color)
        ax.axvline(x=offset, color='black', alpha=0.5)

        for change in change_to_move:
            ax.axvline(t[change], color='green', alpha=0.7)
        for change in change_to_stop:
            ax.axvline(t[change], color='red', alpha=0.7)
        ax.axhline(y=0, color=color, alpha=0.5)
        ax.tick_params(axis='y', labelcolor=color)

        # color = 'tab:blue'
        # ax1 = ax.twinx()
        # ax1.set_ylabel('velocity', color=color)
        # ax1.axhline(y=0, color=color, alpha=0.5)
        # ax1.plot(t, v, color=color)
        # ax1.tick_params(axis='y', labelcolor=color)

        color = 'tab:green'
        ax2 = ax.twinx()
        ax2.set_ylabel('angle', color=color)
        q_low, q_high = angle.quantile(q=(0.25, 0.75))
        ax2.axhline(y=q_low, color=color, alpha=0.5)
        ax2.axhline(y=q_high, color=color, alpha=0.5)
        ax2.axhline(y=+.00, color=color, alpha=0.5)
        ax2.axhline(y=min_angle, color=color, alpha=0.5)
        ax2.set_ylim([0, 181])
        # ax2.axhline(y=+.15, color=color, alpha=0.5)
        # ax2.axhline(y=-.15, color=color, alpha=0.5)
        ax2.plot(t, angle, color=color)
        ax2.tick_params(axis='y', labelcolor=color)

        color = 'blue'
        bx.plot(x, y, color=color)
        bx.scatter(x[start + offset], y[start + offset], color='pink',
                   zorder=0)
        bx.scatter(x_change, y_change, color='orange',  # [start + period + offset]
                   marker='o', facecolors='none', zorder=2)
        bx.scatter(x_stop, y_stop, color='red',  # [start + period + offset]
                   marker='.',  # facecolors='none',
                   zorder=1)
        bx.axis('equal')

    fig.tight_layout()
    logger.debug('Saving single plots: {}'.format(plot_name))
    plt.savefig(plot_name, dpi=300)
    # plt.show()
    plt.close()
    return


def select_tracks(path_to_file=None, daily_directory=None, df=None, fps=None,
                  frame_height=None, frame_width=None, settings=None):
    logger = logging.getLogger('ei').getChild(__name__)
    '''
    settings['verbose']
    settings['path to test .csv']
    settings['frames per second']
    settings['minimal length in seconds']
    settings['force tracking.ini fps settings']
    settings['limit track length to x seconds']
    settings['extreme size outliers lower end in px']
    settings['extreme size outliers upper end in px']
    settings['frame width']
    settings['frame height']
    settings['pixel per micrometre']
    settings['exclude measurement when above x times average area']
    settings['percent quantiles excluded area']
    settings['try to omit motility outliers']
    settings['stop excluding motility outliers if total count above percent']
    settings['limit track length to x seconds']
    settings['limit track length exactly']
    settings['compare angle between n frames']
    settings['minimal angle in degrees for turning point']
    settings['save large plots']
    '''
    if settings is None:
        settings = get_configs()  # Get settings
        if settings is None:
            logger.critical('No settings provided / could not get settings for start_it_up().')
            return None
    if settings['verbose']:
        logger.debug('Have accepted string {}'.format(path_to_file))
    if path_to_file is None:
        path_to_file = settings['path to test .csv']
    if daily_directory is None:
        folder_time = str(strftime('%y%m%d', localtime()))
        daily_directory = '{}/{}_Result_py/'.format(os.path.dirname(path_to_file), folder_time)
        if not os.path.exists(daily_directory):
            try:
                os.makedirs(daily_directory)
                logger.info('Creating daily folder {}'.format(daily_directory))
            except OSError as makedir_error:
                logger.exception(makedir_error)
    file_name = os.path.basename(path_to_file)
    file_name = os.path.splitext(file_name)[0]
    # file_name = datetime.now().strftime('%y%m%d%H%M%S') + '_{}'.format(file_name)

    # Set up and check some basic stuff
    if fps is None or fps <= 0 or settings['force tracking.ini fps settings']:
        if settings['frames per second'] > 0:
            fps = settings['frames per second']
        else:
            logger.critical('fps value is negative or zero; cannot continue.')
            return None
    # change from sec to frames
    settings['minimal length in seconds'] = int(round(fps, 0) * settings['minimal length in seconds'])
    settings['limit track length to x seconds'] = int(round(fps, 0) * settings['limit track length to x seconds'])
    if settings['extreme size outliers lower end in px'] >= settings['extreme size outliers upper end in px']:
        logger.critical('Minimal area exclusion in px^2 larger or equal to maximum; will not be able to find tracks. '
                        'Please update tracking.ini. Min: {}, max: {}'.format(  # makes no sense to continue
                         settings['extreme size outliers lower end in px'],
                         settings['extreme size outliers upper end in px'])
                        )
        return None
    if frame_width is None or frame_height is None:
        logger.debug('Retrieving frame width/height from tracking.ini.')
        frame_width = settings['frame width']
        frame_height = settings['frame height']
    if frame_height <= 0 or frame_width <= 0:
        logger.critical('Frame width or frame height 0 or negative; cannot continue. Width: {}, height: {}'.format(
            frame_width, frame_height
        ))
        return None
    px_to_micrometre = settings['pixel per micrometre']
    if px_to_micrometre <= 0:
        logger.critical('\'pixel per micrometre\' setting in tracking.ini 0 or negative. '
                        'Cannot continue. Value: {}'.format(px_to_micrometre))
        return None
    if type(df) is not pd.core.frame.DataFrame:  # In case we didn't get a data frame
        if settings['verbose']:
            logger.debug('Handing string to get_data {}'.format(path_to_file))
        df = get_data(path_to_file)
    if df is None:  # get_data() returns None in case of errors
        logger.critical('Error reading data frame from file {}'.format(path_to_file))
        return None
    if df.shape[0] < settings['minimal length in seconds']:
        logger.critical('File is empty/of insufficient length before initial clean-up. '
                        'Minimal size (frames): {}, length: {}, path: {}'.format(
                         settings['minimal length in seconds'], df.shape[0], path_to_file))
        return None
    _, track_change = different_tracks(df)  # different_tracks returns [starts], [stops]
    initial_length, initial_size = (len(track_change), df.shape[0])

    df['area'] = df['WIDTH'] * df['HEIGHT']  # calculate area of bacteria
    # In general, area is set to np.NaN if anything is wrong, so we later only have
    # to search for np.NaNs in one column in order to know which rows to remove
    if settings['verbose']:
        logger.debug('Starting to set NaNs')
    # Remove rough outliers
    df['average_area'] = df.groupby('TRACK_ID')['area'].transform('mean')
    df['area'] = np.where(
        (df['average_area'] >= settings['extreme size outliers lower end in px']) &
        (df['average_area'] <= settings['extreme size outliers upper end in px']),
        df['area'],  # track is fine
        np.NaN  # delete otherwise
    )
    # Remove frames where bacterial area is 1.5 times average area (or user defined)
    df['area'] = np.where(
        df['area'] <= (df['average_area'] * settings['exclude measurement when above x times average area']),
        df['area'],  # track is fine
        np.NaN  # delete otherwise
    )
    # set zeroes in area to NaN
    # tracker.py sets width/height as 0 if it can't connect tracks, thus every data point with area == 0 is suspect
    df.loc[df['area'] == 0, 'area'] = np.NaN

    # remove too short frames
    df['length'] = (df.groupby('TRACK_ID')['POSITION_T'].transform('last') -
                    df.groupby('TRACK_ID')['POSITION_T'].transform('first') + 1).astype(np.uint16)
    df['area'] = np.where(
        df['length'] >= settings['minimal length in seconds'],
        df['area'],  # track is fine
        np.NaN  # delete otherwise
    )
    # df['no_move'] = np.sqrt(
    #     (df.groupby('TRACK_ID')['POSITION_X'].transform('last') -
    #      df.groupby('TRACK_ID')['POSITION_X'].transform('first')) ** 2 +
    #     (df.groupby('TRACK_ID')['POSITION_Y'].transform('last') -
    #      df.groupby('TRACK_ID')['POSITION_Y'].transform('first')) ** 2
    # )
    # df['area'] = np.where(
    #     df['no_move'] >= 10,
    #     df['area'],  # track is fine
    #     np.NaN  # delete otherwise
    # )
    # remove all rows with a NaN in them - this gets rid of empty/short tracks and empty/suspect measurements
    # As we'll later need only the remaining areas, we'll drop the NaNs
    if settings['verbose']:
        logger.debug('Dropping NaN values from df')
    df.dropna(inplace=True, subset=['area'])  # df.query('Diff > 0.1')  # df[df.Diff > 0.1]

    # reset index to calculate track_change again
    if settings['verbose']:
        logger.debug('Re-indexing')
    df.reset_index(drop=True, inplace=True)
    if df.shape[0] < settings['minimal length in seconds']:
        logger.warning('File is empty/of insufficient length after initial clean-up. '
                       'Minimal size: {}, length: {}, path: {}'.format(
                        settings['minimal length in seconds'], df.shape[0], path_to_file))
        return None
    track_start, track_change = different_tracks(df)  # as we've dropped rows, this needs to be calculated again
    logger.info(
        'Tracks before initial cleanup: {}, after: {}, loss: {:.4%}, '
        'data frame entries before: {}, after: {}, loss: {:.4%}'.format(
            initial_length, len(track_change), ((initial_length - len(track_change)) / initial_length),
            initial_size, df.shape[0], ((initial_size - df.shape[0]) / initial_size)
        ))

    # get ratio between short side/long side (height divided by width if height is <= width; width/height otherwise)
    # If we'd use long/short ratio, we might have problems with infinite results/division by 0
    df['ratio_wh'] = np.where(df['HEIGHT'] <= df['WIDTH'], df['HEIGHT'] / df['WIDTH'], df['WIDTH'] / df['HEIGHT'])

    # quartiles are used to kick out outlier-ish cases:
    # average track width/height ratio needs to be within quartile range - this necessitates that most measurements are
    # actually bacteria
    if settings['percent quantiles excluded area'] > 0:
        q1_area, q3_area = df['area'].quantile(q=[
            settings['percent quantiles excluded area'], (1 - settings['percent quantiles excluded area'])
        ])
        logger.info('Area 10 % quartiles: {:.2f}, {:.2f}'.format(
            q1_area, q3_area, ))
    else:  # get everything
        q1_area = -1
        q3_area = np.inf
    if settings['try to omit motility outliers']:
        df['distance'] = np.sqrt(np.square(df['POSITION_X'].diff()) +
                                 np.square(df['POSITION_Y'].diff())) / df['POSITION_T'].diff()
        np.put(df['distance'], track_start, 0)
        q1_dist, q3_dist = df['distance'].quantile(q=[0.25, 0.75])  # IQR
        distance_outlier = (q3_dist - q1_dist) * 3 + q3_dist  # outer fence
        df['distance'] = np.where(df['distance'] > distance_outlier, 1, 0).astype(np.int8)
        distance_outlier_percents = df['distance'].sum() / df.shape[0]
        logger.info('25/75 % Distance quartiles: {:.3f}, {:.3f} upper outliers: {:.3f} '
                    'counts: {}, of all entries: {:.4%}'
                    ''.format(q1_dist, q3_dist, distance_outlier, df['distance'].sum(), distance_outlier_percents))

        if distance_outlier_percents > settings['stop excluding motility outliers if total count above percent']:
            logger.warning('Motility outliers more than {:.2%} of all data points ({:.2%}); recommend to '
                           're-analyse file with outlier removal changed if upper quartile is especially low'
                           '(Quartile: {:.3f})'.format(
                            settings['stop excluding motility outliers if total count above percent'],
                            distance_outlier_percents, q3_dist))
            logger.info('Distance outlier exclusion switched off due to too many outliers')
            df['distance'] = np.zeros(df.shape[0], dtype=np.int8)
    else:
        df['distance'] = np.zeros(df.shape[0], dtype=np.int8)

    if settings['verbose']:
        logger.debug('Starting with fine selection')

    # So we can later see why tracks were removed
    kick_reasons = [0 for _ in range(8)]

    # create a list of all tracks that match our selection criteria
    good_track = []

    # Go through all tracks, look for those that match all conditions, append to list
    # @to do: groupby find_good_tracks -> tried that; not possible due to recursive calls / stack overflow
    for start, stop in zip(track_start, track_change):
        good_track_result, kick_reason = find_good_tracks(df_passed=df,
                                                          lower_boundary=q1_area,
                                                          upper_boundary=q3_area,
                                                          start=start,
                                                          stop=stop,
                                                          settings=settings,
                                                          frame_height=frame_height,
                                                          frame_width=frame_width,
                                                          )
        kick_reasons[kick_reason] += 1
        # if good_track_result is empty, skip rest:
        if not good_track_result:
            continue
        # get longest track from good_track_result:
        good_selection = 0
        if len(good_track_result) > 1:
            good_comparator = 0
            for idx_good, (good_start, good_stop) in enumerate(good_track_result):
                curr_length = good_stop - good_start + 1
                if curr_length > good_comparator:
                    good_selection = idx_good
                    good_comparator = curr_length
        good_start, good_stop = good_track_result[good_selection]
        # limit track length
        if settings['limit track length to x seconds']:  # 0 == False
            # Set limit to start time + limit
            limit_track_length_curr = settings['limit track length to x seconds'] + df.loc[good_start, 'POSITION_T'] - 1
            # get index of time point closest to limit or maximum
            if not settings['limit track length exactly']:
                good_stop_curr = df.loc[good_start:good_stop, 'POSITION_T'].where(
                    df.loc[good_start:good_stop, 'POSITION_T'] <= limit_track_length_curr).idxmax()
            else:
                good_stop_curr = df.loc[good_start:good_stop, 'POSITION_T'].where(
                    df.loc[good_start:good_stop, 'POSITION_T'] == limit_track_length_curr).idxmax()
            if np.isnan(good_stop_curr):
                continue
            # if (good_stop_curr - good_start) > settings['limit track length to x seconds']:
            #     logger.debug('Timing off: start: {}, stop: {}, stop calc.:{} diff: {}'.format(
            #         good_start, good_stop, good_stop_curr, (good_stop_curr - good_start)))
            #     continue
            good_stop = good_stop_curr
            # Exclude NaNs in case no index can be found within returned track
        good_track.append((good_start, good_stop))
    logger.debug('All tracks before fine selection: {}, left over: {}, difference: {}'.format(
        len(track_change), len(good_track), (len(track_change) - len(good_track))))
    '''
    # kick_reason: 
    7 size < 600
    6 holes > 6
    5 distance outlier
    4 duration*1.05 > size
    3 average area not within bounds
    2 average ratio not within bounds
    1 average x/y not within bounds
    0 pass
    '''
    kick_reasons_string = 'Total: {8} size < 600: {7} holes > 6: {6} ' \
                          'distance outlier: {5} duration 5% over size: {4} ' \
                          'area out of bounds: {3} ratio wrong: {2} ' \
                          'screen edge: {1} passed: {0} \t{8},'.format(*kick_reasons, sum(kick_reasons))
    for reason in reversed(kick_reasons):  # @todo: remove
        kick_reasons_string += '{},'.format(reason)
    if kick_reasons[0] < 1000 and kick_reasons[0] / sum(kick_reasons) < 0.3:
        logger.warning('Low amount of accepted tracks')
        logger.warning(kick_reasons_string)
    else:
        logger.info(kick_reasons_string)

    if not good_track:  # If we are left with no tracks, we can stop here
        end_string = 'File {} has no acceptable tracks.'.format(path_to_file)
        logger.warning(end_string)
        return end_string
    # Convert good_track to list for use with np.put (a lot quicker than setting slices of np.array to true)
    df['good_track'] = np.zeros(df.shape[0], dtype=np.int8)
    set_good_track_to_true = []
    for (start, stop) in good_track:
        set_good_track_to_true.extend(range(start, (stop + 1), 1))
    np.put(df['good_track'], set_good_track_to_true, 1)
    del set_good_track_to_true

    # Reset df to important bits
    if settings['verbose']:
        logger.debug('Resetting df')
    df = df.loc[df['good_track'] == 1, ['TRACK_ID', 'POSITION_T', 'POSITION_X', 'POSITION_Y', ]]
    df.reset_index(inplace=True)
    diff_tracks_start, track_change = different_tracks(df)

    # set up some overall values
    if settings['verbose']:
        logger.debug('Calculating x_delta, y_delta, t_delta, travelled_dist')
    df['x_delta'] = df['POSITION_X'].diff()
    df['y_delta'] = df['POSITION_Y'].diff()
    df['t_delta'] = df['POSITION_T'].diff()
    # Set correct values for track starts
    np.put(df['x_delta'], diff_tracks_start, 0)
    np.put(df['y_delta'], diff_tracks_start, 0)
    np.put(df['t_delta'], diff_tracks_start, 1)

    for letter in ['x', 'y', 't']:  # validate
        item = '{}_delta'.format(letter)
        if df[item].isnull().any():  # check if any value is still NaN
            logger.critical('{} has NaN value(s) after '
                            'clean-up at position(s): {}'.format(item, np.where(df[item].isnull())[0]))
            logger.critical('{} track starts: {}'.format(item, diff_tracks_start))

    df['POSITION_T'] = df['POSITION_T'].sub(df.groupby('TRACK_ID')['POSITION_T'].transform('first')).astype(np.int32)
    if any(df['POSITION_T'] < 0):  # validate
        logger.critical('POSITION_T contains negative values')
        return None

    if settings['verbose']:
        logger.debug('Starting with statistical calculations per track')

    # travelled_distance = square root(delta_x^2 + delta_y^2) / px to micrometre ratio
    df['travelled_dist'] = np.sqrt(np.square(df['x_delta']) + np.square(df['y_delta'])) / px_to_micrometre
    # np.put(df['travelled_dist'], diff_tracks_start, 0)
    df['minimum'] = df['travelled_dist'] / df['t_delta']
    # get rid of rounding errors, convert to binary:
    df['minimum'] = np.where(df['minimum'] > 10 ** -3, 1, 0).astype(np.int8)
    if int(round(fps, 0)) & 1 == 0:  # if fps is even
        max_kernel = int(round(fps, 0)) + 1
    else:
        max_kernel = int(round(fps, 0))
    # median filter the values to spot general null points in movement
    for kernel_size in [3, max_kernel]:
        df['minimum'] = df.groupby('TRACK_ID')['minimum'].transform(medfilt, kernel_size=kernel_size)
    np.put(df['minimum'], diff_tracks_start, 0)

    angle_diff = settings['compare angle between n frames']
    min_angle = settings['minimal angle in degrees for turning point']
    df['x_diff_track'] = df.groupby('TRACK_ID')['POSITION_X'].diff(angle_diff).fillna(method='bfill')
    df['y_diff_track'] = df.groupby('TRACK_ID')['POSITION_Y'].diff(angle_diff).fillna(method='bfill')
    df['angle_calc'] = np.degrees(np.arctan2(df['x_diff_track'], df['y_diff_track']))
    df['angle_calc'] = abs(df.groupby('TRACK_ID')['angle_calc'].diff().fillna(0))
    df['angle_calc'] = np.where(360 - df['angle_calc'] <= df['angle_calc'], 360 - df['angle_calc'], df['angle_calc'])
    df['turn_points'] = np.where((df['angle_calc'] > min_angle) & (df['minimum'] == 1), 1, 0)

    df['x_norm'] = (df['POSITION_X'].sub(df.groupby('TRACK_ID')['POSITION_X'].transform('first'))) / px_to_micrometre
    df['y_norm'] = (df['POSITION_Y'].sub(df.groupby('TRACK_ID')['POSITION_Y'].transform('first'))) / px_to_micrometre

    # Source: ALollz, stackoverflow.com/questions/51064346/
    # Get largest displacement during track
    pdist_series = df.groupby('TRACK_ID').apply(lambda l: dist.pdist(np.array(list(zip(l.x_norm, l.y_norm)))).max())

    time_series = df.groupby('TRACK_ID')['POSITION_T'].agg('last')
    time_series = (time_series + 1) / fps  # off-by-one error
    dist_series = df.groupby('TRACK_ID')['travelled_dist'].agg('sum')
    motile_total_series = df.groupby('TRACK_ID')['minimum'].agg('sum')
    motile_series = (motile_total_series / df.groupby('TRACK_ID')['POSITION_T'].agg('last'))
    # time, curr_track_null_points_per_s, distance, speed, displacement
    speed_series = pd.Series(
        (np.where(motile_total_series != 0,
                  dist_series / time_series,
                  0)), index=time_series.index)
    turn_percent_series = df.groupby('TRACK_ID')['turn_points'].agg('sum')
    # Avoid div. by 0:
    turn_percent_series = pd.Series(
        (np.where(motile_total_series != 0,
                  turn_percent_series / motile_total_series,
                  0)), index=time_series.index)
    name_of_columns = ['Turn Points (TP/s)',  # 0
                       'Distance (\u00B5m)',  # 1
                       'Speed (\u00B5m/s)',  # 2
                       'Time (s)',  # 3
                       'Displacement',  # 4
                       '% motile',  # 5
                       ]
    # Create df for statistics
    df_stats = pd.concat(
        [turn_percent_series,  # 0
         dist_series,  # 1
         speed_series,  # 2
         time_series,  # 3
         pdist_series,  # 4
         motile_series  # 5
         ],
        keys=name_of_columns, axis=1)
    del turn_percent_series, dist_series, speed_series, time_series, pdist_series, motile_series
    # OH GREAT MOTILITY ORACLE, WHAT WILL MY BACTERIAS MOVES BE LIKE?
    # total count
    all_entries = df_stats.shape[0]
    # @todo: set threshold for Turn Points / s in .ini as well as motility %
    # % likely motile cells according to number of turning points per second
    motile = np.where((df_stats[name_of_columns[0]] < 0.05) &  # 'Turn Points (TP/s)'
                      (df_stats[name_of_columns[5]] > 0.5),  # '% motile'
                      1, 0).sum() / all_entries
    # % likely twitching cells, same metric
    twitching = np.where((df_stats[name_of_columns[0]] > 0.05) &  # 'Turn Points (TP/s)'
                         (df_stats[name_of_columns[5]] > 0.5),  # '% motile'
                         1, 0).sum() / all_entries
    # median, 75 percentile, for motile/immotile decision
    median_perc_motility, q3_perc_motility = df_stats[name_of_columns[5]].quantile(q=(0.5, 0.75))  # '% motile'
    logger.debug('Motile: {} Twitching: {} Median % molility: {} q3 % motility: {}'.format(
        motile, twitching, median_perc_motility, q3_perc_motility  # @todo: fix predictions
    ))
    if median_perc_motility > 0.5:
        if motile >= 2 * twitching:  # low null point/sec fraction is larger
            prediction = 'Motile'
        elif twitching >= 2 * motile:  # high null point/sec fraction is larger
            prediction = 'Twitching'
        else:
            prediction = 'Twitching/Motile'  # No fraction is significantly larger than the other
        if motile + twitching < 0.5:  # indeterminate fraction is largest
            prediction += ', large indeterminate fraction'
    elif q3_perc_motility < 0.5:  # Most cells are immotile
        prediction = 'Immotile'
    else:
        prediction = 'Mixture of motile and immotile cells'

    q1_time, q2_time, q3_time = np.quantile(df_stats[name_of_columns[3]], (0.25, 0.5, 0.75))  # 'Time (s)',  # 3
    logger.debug('Time duration of selected tracks min: {:.3f}, max: '
                 '{:.3f}, Quantiles (25/50/75%): {:.3f}, {:.3f}, {:.3f}'
                 ''.format(min(df_stats[name_of_columns[3]]), max(df_stats[name_of_columns[3]]),
                           q1_time, q2_time, q3_time))

    # Calculate df_stats_seaborne for statistical plots
    name_all_categories = 'All'
    cut_off_list = [(-1, np.inf, name_all_categories),
                    (-1, 0.02, '0 - 0.02'),
                    (0.02, 0.1, '0.02 - 0.1'),
                    # (0.2, 0.3, '0.2 - 0.3'),
                    (0.1, 0.4, '0.1 - 0.4'),
                    (0.4, np.inf, '0.4 +')]

    df_stats['Categories'] = name_all_categories
    df_stats_seaborne = df_stats.copy()
    cut_off_parameter = name_of_columns[0]  # 'Turn Points (TP/s)',  # 0

    # To drop unassigned values later
    df_stats_seaborne['Categories'] = np.NaN
    for index_cut_off, (low, high, category) in enumerate(cut_off_list):
        # Since we'll concatenate df_stats to df_stats_seaborne, which already contains all
        if category == name_all_categories:
            continue
        # inclusive low, exclusive high
        df_stats_seaborne['Categories'] = np.where(
            (low <= df_stats[cut_off_parameter]) &
            (high > df_stats[cut_off_parameter]),
            index_cut_off,
            df_stats_seaborne['Categories']
        )
    # All np.NaN in column get replaced with str when str values are set within column -> easier to exchange later
    df_stats_seaborne.dropna(subset=['Categories'], inplace=True)

    # Exchange int values in 'Categories' for correct labels
    df_stats_seaborne['Categories'].replace(  # name_all_categories is not present and can be skipped
        {value: key for key, value in zip([i for (_, _, i) in cut_off_list[1:]], range(1, len(cut_off_list)))},
        inplace=True)

    df_stats_seaborne = pd.concat([df_stats, df_stats_seaborne], ignore_index=True)

    # Get name of categories, assign values in ascending order, sort df_stats_seaborne['Categories'] by order/dict
    categories = {key: value for key, value in zip([i for (_, _, i) in cut_off_list], range(0, len(cut_off_list)))}
    # sort df_stats_seaborne by generated dict key:value pairs (in order of cut_off_list)
    df_stats_seaborne = df_stats_seaborne.iloc[df_stats_seaborne['Categories'].map(categories).sort_values().index]

    distance_min = df_stats[name_of_columns[1]].min()  # 'Distance (micrometre)',  # 1
    distance_max = df_stats[name_of_columns[1]].max()
    # @todo: make dependent on usage of large plot / rose graph
    df['distance_colour'] = df.groupby('TRACK_ID')['travelled_dist'].transform('sum') - distance_min
    df['distance_colour'] = df['distance_colour'] / df['distance_colour'].max()

    # @todo: allow user customisation of plot title name
    # Set up plot title name
    plot_title_name = file_name.replace('_', ' ')
    plot_title_name = plot_title_name.split('.', 1)[0]
    original_plot_date = plot_title_name[:12]
    # Add time/date to title of plot; convenience for illiterate supervisors
    if original_plot_date.isdigit():
        plot_title_name = plot_title_name[12:]
        try:
            # original_plot_date_as_time_object = strptime(str(original_plot_date), '%y%m%d%H%M%S')
            original_plot_date = strftime('%d. %m. \'%y, %H:%M:%S', strptime(str(original_plot_date), '%y%m%d%H%M%S'))
        except (ValueError, TypeError) as time_val_error:
            logger.exception(time_val_error)
        finally:
            pass
        plot_title_name = '{} {}'.format(original_plot_date, plot_title_name)

    plt.rcParams.update({'font.size': 8})
    plt.rcParams['axes.axisbelow'] = True
    if settings['save large plots']:
        # DIN A4, as used in the civilised world  # @todo: let user select other, less sophisticated, formats
        plt.figure(figsize=(11.6929133858, 8.2677165354))  # , gridspec_kw={'width_ratios': [1, 1, 1, 1]}
        plt.grid(True)
        plt.axis('equal')
        # display initial position
        grouped_df = df.groupby('TRACK_ID')['POSITION_X', 'POSITION_Y'].transform('first')
        plt.scatter(
            grouped_df.POSITION_X,
            grouped_df.POSITION_Y,
            marker='o',
            color='black',
            s=1,
            lw=0,
        )
        grouped_df = df.loc[
                     :, ['TRACK_ID', 'distance_colour', 'POSITION_X', 'POSITION_Y']
                     ].sort_values(['distance_colour'], ascending=False).groupby(
            'TRACK_ID', sort=False)['POSITION_X', 'POSITION_Y', 'distance_colour']
        for name, group in grouped_df:
            plt.scatter(
                group.POSITION_X,
                group.POSITION_Y,
                marker='.',
                label=name,
                c=plt.cm.gist_rainbow(group.distance_colour),
                # vmin=distance_min,
                # vmax=distance_max,
                # cmap=plt.cm.gist_rainbow,
                s=1,
                lw=0,
            )
        del grouped_df
        plt.title('{} Track count: {}'.format(plot_title_name, len(track_change)))
        plot_save_path = '{}{}_Bac_Run_Overview.png'.format(daily_directory, file_name)
        plt.savefig(plot_save_path, dpi=300)
        logger.info('Saving figure {}'.format(plot_save_path))
        plt.close()

    f = plt.figure()
    # DIN A4, as used in the civilised world  # @todo: let user select other, less sophisticated, formats
    f.set_size_inches(11.6929133858, 8.2677165354)
    # @todo: set plot values in .ini?
    outer_space = 0.05
    inner_space = 0.03
    head_space = 0.3
    width_space = 0.05
    gs1 = gridspec.GridSpec(2, 1, figure=f)
    gs1.update(left=outer_space, right=0.5 - inner_space, hspace=head_space, wspace=width_space)
    gs2 = gridspec.GridSpec(2, 100, figure=f)
    gs2.update(left=0.5 + inner_space, right=1 - outer_space, hspace=head_space, wspace=width_space)

    all_plots = [
        plt.subplot(gs1[0, 0]),  # TPs
        plt.subplot(gs2[0, :-2]),  # xy-centered plots
        plt.subplot(gs1[1, 0]),  # Speed
        plt.subplot(gs2[1, :]),  # time
        plt.subplot(gs2[0, -2:]),  # distance color-map
    ]

    ##############################
    if settings['verbose']:
        logger.debug('Setting up plots')
    # So we get a distance range from 0 to 1 for later color selection
    textbox = False
    for plot_idx in range(0, len(all_plots), 1):
        if settings['verbose']:
            logger.debug('Plot {} / {}'.format(plot_idx + 1, len(all_plots)))  # in case plotting takes forever
        if plot_idx == 1:  # 0,0 centered plot of all tracks
            all_plots[plot_idx].set_title('{}\nTracks: {}, Prediction: {}'.format(
                plot_title_name, len(good_track), prediction))

            # get relevant columns, sort by distance (descending), then group sorted df by TRACK_ID
            grouped_df = df.loc[
                         :, ['TRACK_ID', 'distance_colour', 'x_norm', 'y_norm']
                         ].sort_values(['distance_colour'], ascending=False).groupby(
                'TRACK_ID', sort=False)['x_norm', 'y_norm', 'distance_colour']
            # @todo: Circles indicate the mean and 90th percentile net displacements
            for name, group in grouped_df:
                all_plots[plot_idx].scatter(
                    group.x_norm,
                    group.y_norm,
                    marker='.',
                    label=name,
                    c=plt.cm.gist_rainbow(group.distance_colour),
                    # vmin=distance_min,
                    # vmax=distance_max,
                    # cmap=plt.cm.gist_rainbow,
                    s=1,
                    lw=0,
                )
            del grouped_df
            all_plots[plot_idx].set_aspect('equal', adjustable='box')
            all_plots[plot_idx].grid(True)

        elif plot_idx == 4:
            colorbar_map = plt.cm.gist_rainbow
            norm = mpl.colors.Normalize(vmin=distance_min, vmax=distance_max)
            cb = mpl.colorbar.ColorbarBase(all_plots[plot_idx], cmap=colorbar_map, norm=norm, )
            cb.set_label('Distance in \u00B5m')

        elif plot_idx == 3:
            all_plots[plot_idx].grid(axis='y', which='major',
                                     # color='gray',
                                     alpha=0.80, )
            sns.violinplot(y=df_stats_seaborne[name_of_columns[5]],
                           x=df_stats_seaborne['Categories'],
                           # hue=df_stats[name_of_columns[-1]],
                           # dodge=False,
                           orient='v',
                           cut=0,
                           ax=all_plots[plot_idx],
                           scale='count',  # 'width' 'count' 'area'
                           width=0.95,
                           linewidth=1,
                           bw=.2,
                           # inner='stick',
                           )
            # Remove top and right border
            sns.despine(ax=all_plots[plot_idx], offset=0)

            # Empty title so there's 2 lines of space for the text boxes
            all_plots[plot_idx].set_title('\n\n')
            textbox_info = []
            for idx_textbox in range(len(cut_off_list)):
                # Select name of current violin plot x axis
                curr_category = cut_off_list[idx_textbox][2]
                # Select current of current column the subset of current x axis, calculate median and mean
                curr_entries = sum(df_stats_seaborne['Categories'] == curr_category)
                # logger.info('Category: {} Counts: {}'.format(curr_category, curr_entries))
                df_stats_subset = df_stats_seaborne.loc[df_stats_seaborne['Categories'] == curr_category,
                                                        name_of_columns[5]]
                # Calculate median and average of current violin plot
                qm_plot = df_stats_subset.median()
                average_plot = df_stats_subset.mean()
                if np.isnan(qm_plot):
                    continue
                if all_entries > 0:
                    curr_percentage = curr_entries / all_entries
                else:
                    curr_percentage = 'error'
                textbox_info.append((curr_category, curr_percentage, qm_plot, average_plot))

            for idx_textbox, (curr_category, curr_percentage, qm_plot, average_plot) in enumerate(textbox_info):
                all_plots[plot_idx].text(idx_textbox / len(textbox_info) + 0.015, 1.005,
                                         '{}: {:.1%}\nMedian: {:.2%}\nAverage:  {:.2%}'.format(
                                             curr_category, curr_percentage,
                                             qm_plot,
                                             average_plot),
                                         # Set Textbox to relative position instead of absolute xy coordinates (0-1)
                                         transform=all_plots[plot_idx].transAxes,
                                         )

        else:  # statistical plot
            textbox = True
            all_plots[plot_idx].grid(axis='y', which='major',
                                     # color='gray',
                                     alpha=0.80, )

            sns.violinplot(y=df_stats_seaborne[name_of_columns[plot_idx]],
                           x=df_stats_seaborne['Categories'],
                           # hue=df_stats[name_of_columns[-1]],
                           # dodge=False,
                           orient='v',
                           cut=0,
                           ax=all_plots[plot_idx],
                           scale='count',  # 'width' 'count' 'area'
                           width=0.95,
                           linewidth=1,
                           bw=.2,
                           # inner='stick',
                           )
            # Remove top and right border
            sns.despine(ax=all_plots[plot_idx], offset=0)

            # Empty title so there's 2 lines of space for the text boxes
            all_plots[plot_idx].set_title('\n\n')

            # Create description (title) for each violin plot
        if textbox:
            textbox = False
            # for idx_textbox in range(text_box_count):
            #   (curr_category, curr_percentage, qm_plot, average_plot
            textbox_info = []
            for idx_textbox in range(len(cut_off_list)):
                # Select name of current violin plot x axis
                curr_category = cut_off_list[idx_textbox][2]
                # Select current of current column the subset of current x axis, calculate median and mean
                curr_entries = sum(df_stats_seaborne['Categories'] == curr_category)
                # logger.info('Category: {} Counts: {}'.format(curr_category, curr_entries))
                df_stats_subset = df_stats_seaborne.loc[df_stats_seaborne['Categories'] == curr_category,
                                                        name_of_columns[plot_idx]]
                # Calculate median and average of current violin plot
                qm_plot = df_stats_subset.median()
                average_plot = df_stats_subset.mean()
                if np.isnan(qm_plot):
                    continue
                if all_entries > 0:
                    curr_percentage = curr_entries / all_entries
                else:
                    curr_percentage = 'error'
                textbox_info.append((curr_category, curr_percentage, qm_plot, average_plot))

            for idx_textbox, (curr_category, curr_percentage, qm_plot, average_plot) in enumerate(textbox_info):
                all_plots[plot_idx].text(idx_textbox / len(textbox_info) + 0.015, 1.005,
                                         '{}: {:.1%}\nMedian: {:.2f}\nAverage:  {:.2f}'.format(
                                             curr_category, curr_percentage,
                                             qm_plot,
                                             average_plot),
                                         # Set Textbox to relative position instead of absolute xy coordinates (0-1)
                                         transform=all_plots[plot_idx].transAxes,
                                         )
            # Setting limit is done automatically anyway, doesn't change anything
            # if plot_idx == 0:
            #     all_plots[plot_idx].set_ylim(top=2.5, bottom=None, auto=True)
            # Forsooth, this be a shite solution! @todo: plot min/max in .ini
            # if plot_idx == 0:
            #     all_plots[plot_idx].set_ylim([0, 1.5])
            # elif plot_idx == 1:
            #     all_plots[plot_idx].set_ylim([0, 350])
            # if plot_idx == 2:
            #     all_plots[plot_idx].set_ylim([0, 17.5])
            # elif plot_idx == 3:
            #     all_plots[plot_idx].set_ylim([0, 80])
            # Create violin plot on current axis
        # Manually rename x axis ticks on all subplots
        # for axis in range(len(all_plots)):
        #     all_plots[axis].set_xticklabels(['All', '0 - 25', '25 - 50', '50 - 75', '75 - 100'])
    # tight_layout() clashes with plt.subplots(w, h, constrained_layout=True), which workes better anyway.
    # plt.tight_layout()
    if settings['verbose']:
        logger.debug('Saving figure {}'.format('{}{}_Bac_Run_Statistics.png'.format(daily_directory, file_name)))
    plt.savefig('{}{}_Bac_Run_Statistics.png'.format(daily_directory, file_name), dpi=300)
    logger.info('Statistics picture: {}{}_Bac_Run_Statistics.png'.format(daily_directory, file_name))
    plt.close()
    # slices = np.r_[slice(list_of_selected_plots[0][0], list_of_selected_plots[0][1] + 1),
    #                slice(list_of_selected_plots[1][0], list_of_selected_plots[1][1] + 1),
    #                slice(list_of_selected_plots[2][0], list_of_selected_plots[2][1] + 1),
    #                slice(list_of_selected_plots[3][0], list_of_selected_plots[3][1] + 1),
    #                slice(list_of_selected_plots[4][0], list_of_selected_plots[4][1] + 1),
    #                slice(list_of_selected_plots[5][0], list_of_selected_plots[5][1] + 1)]
    # with open('{}{}_df.csv'.format(daily_directory, file_name), 'w+', newline='\n') as csv_file:
    #     df.iloc[slices, :].to_csv(csv_file)
    # if False:
    #     df_single_plots = df.iloc[slices, :].copy()
    #     sexy_single_plots_in_your_area(
    #         df=df_single_plots,
    #         # list_of_selected_plots=list_of_selected_plots,
    #         px_to_micrometre=px_to_micrometre,
    #         # period=period,
    #         # avg_mask=avg_mask,
    #         plot_name='{}{}_single_tracks.png'.format(daily_directory, file_name),
    #         fps=fps)
    # else:
    #     logger.info('No single tracks selected.')
    end_string = 'Done evaluating file {}'.format(file_name)
    logging.info(end_string)
    return end_string


def start_it_up(path_to_files, df=None, fps=None, frame_height=None, frame_width=None, daily_directory=None,
                settings=None, create_logger=True, ):
    logger = logging.getLogger('ei').getChild(__name__)
    '''
    settings['log_level']
    settings['log file path']
    settings['shorten displayed logging output']
    settings['shorten logfile logging output']
    settings['log to file']
    '''
    if settings is None:
        settings = get_configs()
        if settings is None:
            logger.critical('No settings provided / could not get settings for start_it_up().')
            return None
    if create_logger:
        get_loggers(log_level=settings['log_level'],
                    logfile_name=settings['log file path'],
                    short_stream_output=settings['shorten displayed logging output'],
                    short_file_output=settings['shorten logfile logging output'],
                    log_to_file=settings['log to file'],)
    end_string = None
    folder_time = str(strftime('%y%m%d', localtime()))
    dir_form = '{}/{}_Results/'  # @todo: specify results folder
    if daily_directory is None:
        if isinstance(path_to_files, str) or isinstance(path_to_files, os.PathLike):
            daily_directory = dir_form.format(os.path.dirname(path_to_files), folder_time)
        elif isinstance(path_to_files, list) or isinstance(path_to_files, tuple):
            daily_directory = dir_form.format(os.path.dirname(path_to_files[0]), folder_time)
        else:
            daily_directory = dir_form.format('.', folder_time)
            logger.critical('Could not access base path in path to files; '
                            'results folder set to {}'.format(os.path.abspath(daily_directory)))
        if not os.path.exists(daily_directory):
            try:
                _mkdir(daily_directory)
                logger.info('Results folder: {}'.format(daily_directory))
            except OSError as makedir_error:
                logger.exception(makedir_error)
                logger.warning('Unable to create {}, Directory changed to {}'.format(
                    daily_directory, os.path.abspath('./')))
                daily_directory = './'
            finally:
                pass
    # Check type of path_to_files
    if isinstance(path_to_files, str) or isinstance(path_to_files, os.PathLike):
        # Skip list read via get_data() if we have a data frame:
        if type(df) is pd.core.frame.DataFrame:
            logger.debug('Passing data frame to select_tracks(): {}'.format(path_to_files))
            end_string = select_tracks(path_to_file=path_to_files,
                                       daily_directory=daily_directory,
                                       df=df,
                                       fps=fps,
                                       frame_height=frame_height,
                                       frame_width=frame_width,
                                       settings=settings)
        # Proceed normally otherwise:
        elif os.path.isfile(path_to_files):
            logger.debug('Passing string to select_tracks(): {}'.format(path_to_files))
            end_string = select_tracks(path_to_file=path_to_files,
                                       daily_directory=daily_directory,
                                       fps=fps,
                                       frame_height=frame_height,
                                       frame_width=frame_width,
                                       settings=settings)
        else:
            logger.warning('File {} was skipped during evaluation, '
                           'file did not exist or could not be accessed. '.format(path_to_files))
    # If we got an iterable, go through each entry:
    elif isinstance(path_to_files, list) or isinstance(path_to_files, tuple):
        end_string = []
        for curr_path_to_file in path_to_files:
            if os.path.isfile(curr_path_to_file):
                logger.debug('Passing string to select_tracks(): {}'.format(path_to_files))
                end_string.append(select_tracks(path_to_file=path_to_files,
                                                daily_directory=daily_directory,
                                                fps=fps,
                                                frame_height=frame_height,
                                                frame_width=frame_width,
                                                settings=settings))
            else:
                logger.warning('File {} was skipped during evaluation, '
                               'file did not exist or could not be accessed. '.format(curr_path_to_file))
    else:
        end_string = 'Passed wrong argument(s) to start_it_up(): {}, ' \
                     'should be string/path or list of strings/paths. Argument: ' \
                     '{}'.format(type(path_to_files), path_to_files)
        logger.critical(end_string)
    return end_string


if __name__ == '__main__':
    t_one = datetime.now()  # to get rough time estimation
    # Log message setup
    settings_ = get_configs()  # Get settings
    if settings_ is None:
        sys.exit('Fatal error in retrieving tracking.ini')
    _backup()
    queue_listener, format_for_logging = get_loggers(
        log_level=settings_['log_level'],
        logfile_name=settings_['log file path'],
        short_stream_output=settings_['shorten displayed logging output'],
        short_file_output=settings_['shorten logfile logging output'],
        log_to_file=settings_['log to file'],
    )
    # Log some general stuff
    logger_main = logging.getLogger('ei').getChild(__name__)
    explain_logger_setup = format_for_logging.format(**{
        'asctime': 'YYYY-MM-DD HH:MM:SS,mmm',  # ISO8601 'YYYY-MM-DD HH:MM:SS+/-TZ'
        'name': 'logger name',
        'funcName': 'function name',
        'lineno': 'lNr',
        'levelname': 'level',
        'process': 'PID',
        'message': 'Message (lNr: line number, PID: Process ID)'
    })
    filler_for_logger = ''  # Stupid tabs
    for sub_string in explain_logger_setup.split('\t'):  # create filler with '#' and correct tab placement
        filler_for_logger += '#' * len(sub_string) + '\t'
    filler_for_logger = filler_for_logger[:-1]  # remove last tab
    logger_main.debug('Logging test message')

    pool = mp.Pool()
    main_files = find_paths(base_path='H:/Test/190430_Motility/test/',
                            extension='list.csv', minimal_age=0)
    main_folder_time = str(strftime('%y%m%d', localtime()))
    main_dir_form = '{}/{}_Results/'
    if isinstance(main_files, str) or isinstance(main_files, os.PathLike):
        main_daily_directory = main_dir_form.format(os.path.dirname(main_files), main_folder_time)
    elif isinstance(main_files, list) or isinstance(main_files, tuple):
        main_daily_directory = main_dir_form.format(os.path.dirname(main_files[0]), main_folder_time)
    else:
        logger_main.critical('Could not access base path in path to files; '
                             'results folder set to {}'.format(os.path.abspath('./')))
        main_daily_directory = main_dir_form.format('.', main_folder_time)
    if not os.path.exists(main_daily_directory):
        try:
            os.makedirs(main_daily_directory)
            logger_main.info('Results folder: {}'.format(main_daily_directory))
        except OSError as mkdir_error:
            logger_main.exception(mkdir_error)
            logger_main.warning('Unable to create {}, Directory changed to {}'.format(
                main_daily_directory, os.path.abspath('./')))
            main_daily_directory = './'
        finally:
            pass
    logger_main.info('Paths: {}'.format(len(main_files)))

    for d in main_files:
        pool.apply_async(start_it_up, args=(d,))
    pool.close()
    pool.join()
    logger_main.debug('Elapsed time: {}'.format(elapsed_time(t_one)))
    queue_listener.stop()
