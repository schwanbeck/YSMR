#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Copyright 2019, 2020 Julian Schwanbeck (julian.schwanbeck@med.uni-goettingen.de)
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
import os
from time import sleep
from time import strftime, strptime

import cv2
import numpy as np
import pandas as pd
from scipy.ndimage import binary_propagation
from scipy.signal import medfilt
from scipy.spatial import distance as dist

from ysmr.helper_file import (argrelextrema_groupby, create_results_folder, different_tracks, get_configs, get_data,
                              get_loggers, reshape_result, save_df_to_csv, save_list, sort_list)
from ysmr.plot_functions import angle_distribution_plot, large_xy_plot, rose_graph, violin_plot
from ysmr.tracker import CentroidTracker

__all__ = ['track_bacteria', 'select_tracks', 'evaluate_tracks', 'annotate_video']


def track_bacteria(video_path, settings=None, result_folder=None):
    """
    Detect and track bright spots in a video file, save output to a .csv file

    :param video_path: path to video file
    :param settings: settings from tracking.ini, will be read if not provided
    :type settings: dict
    :param result_folder: path to result folder
    :return: pandas data frame with results, fps of file, frame width, frame height, path to .csv file
    """
    logger = logging.getLogger('ysmr').getChild(__name__)
    settings = get_configs(settings)  # Get settings
    if settings is None:
        logger.critical('No settings provided / could not get settings for start_it_up().')
        return None
    # We may have to set the log level/loggers again due to multiprocessing
    get_loggers(
        log_level=settings['log_level'],
        logfile_name=settings['log file path'],
        short_stream_output=settings['shorten displayed logging output'],
        short_file_output=settings['shorten logfile logging output'],
        log_to_file=settings['log to file'])
    # Check for errors
    if not os.path.isfile(video_path):
        logger.critical('File {} does not exist'.format(video_path))
        return None
    try:
        cap = cv2.VideoCapture(video_path)
    except (IOError, OSError) as io_error:
        logger.exception('Cannot open file {} due to error: {}'.format(video_path, io_error))
        return None

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count < settings['minimal frame count']:
        logger.warning('File {} too short; file was skipped. Limit for \'minimal frame count\': {}'.format(
            video_path, settings['minimal frame count']))
        return None
    if not settings['force tracking.ini fps settings']:
        try:
            fps_of_file = cap.get(cv2.CAP_PROP_FPS)
            if settings['verbose'] or fps_of_file != settings['frames per second']:
                logger.info('fps of file: {}'.format(fps_of_file))
        except Exception as ex:
            template = 'An exception of type {0} occurred while accessing fps from file {2}. Arguments:\n{1!r}'
            logger.exception(template.format(type(ex).__name__, ex.args, video_path))
            if settings['frames per second'] <= 0:
                logger.critical('User defined fps unacceptable: type: {} value: {}'.format(
                    type(settings['frames per second']), settings['frames per second']))
                return None
            else:
                fps_of_file = settings['frames per second']
    else:
        fps_of_file = settings['frames per second']

    if not result_folder:
        result_folder = create_results_folder(video_path)

    pathname, filename_ext = os.path.split(video_path)
    filename = os.path.splitext(filename_ext)[0]
    logger.info('Starting with file {}'.format(video_path))

    # Set initial values; initialise result list
    old_list, list_name = save_list(path=video_path,
                                    result_folder=result_folder,
                                    first_call=True,
                                    rename_old_list=settings['rename previous result .csv'],
                                    illumination=settings['include luminosity in tracking calculation'])
    # Save old_list_name for later if it exists; False otherwise
    ct = CentroidTracker(  # Initialise tracker instance
        max_disappeared=fps_of_file,
        use_gsff=not settings['disable gsff'],
        fps=fps_of_file,
        n_min=settings['minimum horizon size'],
        n_max=settings['maximum horizon size'],
        n_f=settings['number of LSFFs'],
    )

    coords = []  # Empty list to store calculated coordinates
    curr_frame_count = 0
    threshold_list = []
    # skip_frames = 0
    fps_total = []  # List of calculated fps
    error_during_read = False  # Set to true if some errors occur; used to restore old list afterwards if it exists
    (objects, degrees) = (None, None)  # reset objects, additional_info (caused errors in the past)

    # Wether the result of the threshold should convert from black to white on the inverse
    if settings['white bacteria on dark background']:
        threshold_type = cv2.THRESH_BINARY
    else:
        # Simply inverse output (as above)
        threshold_type = cv2.THRESH_BINARY_INV
        settings['threshold offset for detection'] = (settings['threshold offset for detection'] * -1)

    if settings['debugging'] and settings['display video analysis']:
        # Display first frame in case frame-by-frame analysis is necessary
        ret, frame = cap.read()
        cv2.imshow('{}'.format(filename_ext), frame)
    (frame_height, frame_width) = (int(cap.get(4)),
                                   int(cap.get(3)))  # Image dimensions
    if settings['verbose']:
        logger.debug('Frame height: {}, width: {}'.format(frame_height, frame_width))

    # Background removal:
    # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    # if settings['save video']:
    #     output_video_name = '{}/{}_output.avi'.format(result_folder, filename)
    #     logger.info('Output video file: {}'.format(output_video_name))
    #     out = cv2.VideoWriter(output_video_name,
    #                           cv2.VideoWriter_fourcc(*'MJPG'),  # Codec *'MJPG'  # X264
    #                           fps_of_file,  # FPS
    #                           (frame_width, frame_height)  # Dimensions
    #                           )
    # # min_frame_count += skip_frames

    while True:  # Loop over video
        # if cv2.waitKey(1) & 0xFF == ord('n'):  # frame-by-frame
        timer = cv2.getTickCount()
        ret, frame = cap.read()
        # ret: True/False, depends on whether another frame could be retrieved
        # frame: the actual current frame

        # if curr_frame_count < skip_frames:
        #     continue  # skip frame/jump back to start
        # uMatframe = cv2.UMat(frame)
        # UMat: should utilise graphics card; tends to slow down the whole thing a lot
        # gray = cv2.UMat(gray)  # Put after gray conversion

        # Stop conditions
        if not ret and (frame_count == curr_frame_count + 1 or  # some file formats skip one frame
                        frame_count == curr_frame_count) and frame_count >= settings['minimal frame count']:
            # If a frame could not be retrieved and the minimum frame nr. has been reached
            logger.debug('Frames from file {} read.'.format(filename_ext))
            break
        elif not ret:  # Something must've happened, user decides if to proceed
            logger.critical('Error during cap.read() with file {}'.format(video_path))
            error_during_read = settings['stop evaluation on error']
            break

        gray = cv2.cvtColor(frame, settings['color filter'])  # Convert to gray scale

        blurred = cv2.GaussianBlur(gray, (3, 3), 0)  # blur

        # All pixels above curr_threshold are set to 255 (white); others are set to 0
        if settings['adaptive double threshold'] >= 0:
            # @todo: set kernel size as option
            kernel_for_thresholding = 11
            # Using adaptive double threshold
            thresh = cv2.adaptiveThreshold(
                blurred,  # src=
                255,  # maxValue=
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,  # adaptiveMethod=
                threshold_type,  # thresholdType=
                kernel_for_thresholding,  # blockSize=
                # If the negative is taken it correlates best with the same settings for non-adaptive double threshold
                (settings['threshold offset for detection'] * -1),  # C= (Offset)
            )
            if settings['adaptive double threshold'] > 0:
                # as above
                markers_for_thresh = cv2.adaptiveThreshold(
                    blurred,
                    255,
                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    threshold_type,
                    kernel_for_thresholding,
                    ((settings['threshold offset for detection'] +
                      settings['adaptive double threshold']) * -1)
                )
                if settings['debugging'] and settings['display video analysis']:
                    cv2.imshow('Adaptive double threshold markers', markers_for_thresh)
                thresh = binary_propagation(
                    markers_for_thresh,
                    mask=thresh
                ).astype(np.uint8) * 255

        else:
            # Using average gray value of image
            mean, stddev = cv2.meanStdDev(gray)
            if settings['white bacteria on dark background']:
                curr_frame_threshold = (mean + stddev + settings['threshold offset for detection'])
                # total_threshold += curr_frame_threshold
                # Bacteria are brighter than background
            else:
                curr_frame_threshold = (mean - stddev - settings['threshold offset for detection'])
                # total_threshold += curr_frame_threshold
                # Bacteria are darker than background
                # It's sadly not simply (255 - threshold)
            # Non-moving-average version:
            # curr_threshold = int(total_threshold / (curr_frame_count + 1))  # average input  - skip_frames

            # 5 s moving average:
            threshold_list.append(curr_frame_threshold)
            curr_threshold = int(sum(threshold_list) / len(threshold_list))
            if len(threshold_list) > fps_of_file * 5:
                del threshold_list[0]

            if curr_frame_count == settings['minimal frame count']:
                logger.debug(
                    'Background threshold level: {} (of 255), '
                    'mean: {:.2f}, std. deviation: {:.2f}, offset: {}'.format(
                        curr_threshold, mean.item(), stddev.item(), settings['threshold offset for detection']
                    )
                )

            thresh = cv2.threshold(
                blurred,  # src=
                curr_threshold,  # thresh=
                255,  # maxval=
                threshold_type,  # type=
            )[1]

        # Other threshold variations; proved unnecessary:
        # thresh = cv2.threshold(blurred, 90, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]

        # various other tries to optimise threshold:
        # blurred = cv2.bilateralFilter(gray, 3, 75, 75)
        # equ = clahe.apply(gray)  # uncomment clahe above; background removal
        # blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        # blurred = cv2.medianBlur(gray, 5)

        # display individual conversion steps to see which one acts up
        if settings['debugging'] and settings['display video analysis']:
            # cv2.imshow('frame', frame)
            # cv2.imshow('gray', gray)
            # cv2.imshow('equ', equ)
            # cv2.imshow('blurred', blurred)
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
            # cv2.minAreaRect returns ((x, y), (w, h), degrees)

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
                # tracker.py has been changed to adaptively take n dimensions for distance matrix
            else:
                rects.append(reshape_result(reichtangle))
                # reshape_result(tuple_of_tuples) returns ((x, y[, *args]), (w, h, degrees_orientation))

            if settings['display video analysis']:  # or settings['save video']:  # Display bounding boxes
                box = np.int0(cv2.boxPoints(reichtangle))
                # Box: 4 x/y coordinates
                cv2.drawContours(frame, [box], -1, (255, 0, 0), 0)

        objects, wh_degrees = ct.update(rects)  # Calls CentroidTracker.update() from tracker.py

        for index, (objectID, centroid) in enumerate(objects.items()):  # object.items() loop
            # Follow the KISS principle (fancy smoother option is surely available, but this works):
            # Append results to list (Frame, ID, x, y, (other values)), save list when it gets too long
            coords.append((curr_frame_count, objectID, centroid, wh_degrees[objectID]))

            # draw both the ID of the object and the center point
            if settings['display video analysis']:  # or settings['save video']:  # and objectID == curr_bac:
                text = '{}'.format(objectID)
                # Display object ID:
                cv2.putText(frame, text, (int(centroid[0]) - 10, int(centroid[1]) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 0)
                # Display centroid:
                cv2.circle(frame, (int(centroid[0]), int(centroid[1])), 0, (0, 255, 0), -1)

        # # Frame is finished, video can be saved
        # if settings['save video']:
        #     cv2.putText(frame,  # image
        #                 '{}'.format(filename[:].replace('_', ' ')),  # text
        #                 (20, 20),  # xy coordinates
        #                 cv2.FONT_HERSHEY_SIMPLEX,  # font
        #                 0.7,  # text size
        #                 (220, 220, 60),  # colour
        #                 1  # line thickness
        #                 )
        #     out.write(frame)

        # Change coords.list if it is long enough (I/O-operations are time consuming)
        if len(coords) >= settings['list save length interval']:
            # shift this block into previous for-loop if too many objects per frame causes problems
            # change save length value in tracking.ini if current value is an issue.
            # send coords off to be saved on drive:
            save_list(coords=coords, path=list_name,
                      illumination=settings['include luminosity in tracking calculation'])
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
            cv2.imshow('{} unfiltered possible detections'.format(filename_ext), frame)  # Display the image
            if cv2.waitKey(1) & 0xFF == ord('q'):  # Interrupt display on 'q'-keypress
                error_during_read = True
                logger.error('Processing file interrupted by user: {}'.format(video_path))
                break

    if coords:  # check if list is not empty ([] == False, otherwise True)
        save_list(coords=coords, path=list_name,
                  illumination=settings['include luminosity in tracking calculation'])  # Save the remainder

    # if settings['save video']:
    #     out.release()
    if settings['display video analysis']:
        cv2.destroyAllWindows()  # Close active windows
    cap.release()

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
    try:
        last_object_id = next(reversed(objects))  # get number of last object
    except StopIteration:
        logger.warning('Did not track any objects. File: {}'.format(video_path))
        return None
    df_for_eval = sort_list(file_path=list_name, save_file=not settings['delete .csv file after analysis'])

    logger.info('Average frames analysed per second: {}, objects: {}, frames: {}, csv: {}'.format(  # Display some infos
        '{:.2f}'.format((sum(fps_total) / curr_frame_count)).rjust(6, ' '),  # Average FPS
        '{}'.format(last_object_id + 1).rjust(6, ' '),  # Total Nr. of objects
        '{:>6} of {:>6}'.format(curr_frame_count, frame_count),  # Current frames / total frames
        list_name  # results
    ))

    if error_during_read:
        logger.critical('Error during read, stopping before evaluation. File: {}'.format(video_path))
        return None
    return df_for_eval, fps_of_file, frame_height, frame_width, list_name


def find_good_tracks(df_passed, start, stop, lower_boundary, upper_boundary, frame_height,
                     frame_width, settings, minimal_length_frames, recursion=0):
    """
    checks multiple attributes for passed track, returns list with
    ok start/stop indexes and lowest reached kick reason

    :param df_passed: pandas data frame with tracks
    :param start: start index of track
    :type start: int
    :param stop: stop index of track
    :type stop: int
    :param lower_boundary: low bacterial size boundary
    :type lower_boundary: float
    :param upper_boundary: high bacterial size boundary
    :type upper_boundary: float
    :param frame_height: height of frame
    :type frame_height: int
    :param frame_width: width of frame
    :type frame_width: int
    :param settings: tracking.ini settings
    :type settings: dict
    :param minimal_length_frames: minimal length of track in frames
    :type minimal_length_frames: int
    :param recursion: internal recursion level in case of split tracks
    :return: list of (start, stop) indices that passed and lowest kick reason
    :rtype return_result: list
    :rtype kick_reason: int
    """
    logger = logging.getLogger('ysmr').getChild(__name__)
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
    '''
    kick_reason = 7
    return_result = []
    sub_part = []
    # Too short tracks aren't useful and can immediately be discarded
    if size >= minimal_length_frames:  # as corrected to frames from fps * s
        # Do not allow tracking holes for more than n frames, try to find useful halves otherwise
        kick_reason -= 1
        # We keep df_passed for later
        df = df_passed.iloc[start:stop + 1]  # iloc upper range in slice is exclusive
        size = df.shape[0]
        look_for_holes = df['POSITION_T'].diff()
        if look_for_holes.max() <= settings['maximal consecutive holes']:
            kick_reason -= 1
            # Check if there are no outliers (at pos. 0 would be ok)
            if df['distance'].sum() == 0:
                kick_reason -= 1
                # remove tracks with too big holes (>=n%)
                # calculate difference between first and last frame
                duration = df['POSITION_T'].iloc[-1] - df['POSITION_T'].iloc[0] + 1  # avoid off-by-one error
                duration_size_ratio = duration / size
                if duration_size_ratio < settings['maximal empty frames in %']:
                    kick_reason -= 1
                    # Average area should be within 0.1/0.9 quartile
                    # off all detections if mostly bacteria are detected
                    if lower_boundary <= df['area'].mean() <= upper_boundary:
                        kick_reason -= 1
                        # width/height ratio is bacterial shape specific
                        if (settings['average width/height ratio min.']
                                < df['ratio_wh'].mean()
                                < settings['average width/height ratio max.']):
                            kick_reason -= 1
                            # exclude n% of screen edges
                            if (settings['percent of screen edges to exclude'] * frame_height
                                    < df['POSITION_Y'].mean()
                                    < (1 - settings['percent of screen edges to exclude']) * frame_height):
                                if (settings['percent of screen edges to exclude'] * frame_width
                                        < df['POSITION_X'].mean()
                                        < (1 - settings['percent of screen edges to exclude']) * frame_width):
                                    kick_reason -= 1
                                    # everything is as it should be, append start/stop to return_result
                                    return_result.append((start, stop))
            # Split track into part before/after outlier; try to analyze those instead; extend return_result
            else:
                # Hole is index where distance_outlier == 1; hole is excluded
                idx_outlier = df['distance'].idxmax()
                sub_part.extend([(start, idx_outlier - 1), (idx_outlier + 1, stop)])
        # Split track into part before/after hole; try to analyze those instead; extend return_result
        else:
            # Hole is index of largest number in df['POSITION_T'].diff()
            idx_hole = look_for_holes.idxmax()
            sub_part.extend([(start, idx_hole - 1), (idx_hole, stop)])
    elif recursion >= settings['maximal recursion depth'] != 0:
        logger.debug('Recursion reached max. level at TRACK_ID: {} start: {} stop: {}'.format(
            df_passed.loc[start, 'TRACK_ID'], start, stop))
    # stop recursions from causing stack overflows
    if sub_part and recursion < settings['maximal recursion depth']:
        kick_reason_list = [kick_reason]
        for (sub_start, sub_stop) in sub_part:
            # Skip to stop unnecessary recursions; < 3 in case of minimal_length_frames == 0
            if minimal_length_frames < 3:
                if sub_stop - sub_start + 1 < 3:
                    continue
            elif sub_stop - sub_start + 1 < minimal_length_frames:
                continue
            sub_return_result, kick_reason = find_good_tracks(
                df_passed=df_passed,
                lower_boundary=lower_boundary,
                upper_boundary=upper_boundary,
                settings=settings,
                start=sub_start,
                stop=sub_stop,
                frame_height=frame_height,
                frame_width=frame_width,
                minimal_length_frames=minimal_length_frames,
                recursion=recursion + 1,
            )
            kick_reason_list.append(kick_reason)
            return_result.extend(sub_return_result)
        # Get the smallest kick reason (default kick_reason, but sub-part might've gotten further)
        kick_reason = min(kick_reason_list)
    return return_result, kick_reason


def select_tracks(path_to_file=None, df=None, results_directory=None, fps=None,
                  frame_height=None, frame_width=None, settings=None, **_):
    """
    selection of good tracks from file or data frame according to various parameters
    either file or data frame have to be provided

    :param path_to_file: optional path to .csv
    :param df: optional pandas data frame
    :param results_directory: path to results directory
    :param fps: frame per second value
    :type fps: float
    :param frame_height: frame height
    :rtype frame_height: int
    :param frame_width: frame width
    :rtype frame_width: int
    :param settings: tracking.ini settings
    :type settings: dict
    :return: pandas data frame of selected tracks
    """
    # @todo: use kwargs to overwrite settings dict
    logger = logging.getLogger('ysmr').getChild(__name__)
    settings = get_configs(settings)  # Get settings
    if settings is None:
        logger.critical('No settings provided / could not get settings for start_it_up().')
        return None
    if settings['verbose']:
        logger.debug('Have accepted string {}'.format(path_to_file))
    if path_to_file is None:
        path_to_file = settings['path to test .csv']
    if results_directory is None:
        results_directory = create_results_folder(path_to_file)
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
    minimal_length_frames = int(round(fps, 0) * settings['minimal length in seconds'])
    limit_track_to_frames = int(round(fps, 0) * settings['limit track length to x seconds'])
    if settings['extreme area outliers lower end in px*px'] >= settings['extreme area outliers upper end in px*px']:
        logger.critical(
            'Minimal area exclusion in px^2 larger or equal to maximum; will not be able to find tracks. '
            'Please update tracking.ini. extreme area outliers lower end in px*px: {}, '
            'extreme area outliers upper end in px*px: {}'.format(  # makes no sense to continue
                settings['extreme area outliers lower end in px*px'],
                settings['extreme area outliers upper end in px*px']
            )
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
    if df.shape[0] < minimal_length_frames:
        logger.critical(
            'File is empty/of insufficient length before initial clean-up. '
            'Minimal size (frames): {}, length: {}, path: {}'.format(
                minimal_length_frames, df.shape[0], path_to_file
            )
        )
        return None
    _, track_change = different_tracks(df)  # different_tracks returns [starts], [stops]
    initial_length, initial_size = (len(track_change), df.shape[0])

    df['area'] = df['WIDTH'] * df['HEIGHT']  # calculate area of bacteria in px**2
    # In general, area is set to np.NaN if anything is wrong, so we later only have
    # to search for np.NaNs in one column in order to know which rows to remove
    if settings['verbose']:
        logger.debug('Starting to set NaNs')
    # Remove rough outliers
    df['average_area'] = df.groupby('TRACK_ID')['area'].transform('mean')
    df['area'] = np.where(
        (df['average_area'] >= settings['extreme area outliers lower end in px*px']) &
        (df['average_area'] <= settings['extreme area outliers upper end in px*px']),
        df['area'],  # track is fine
        np.NaN  # delete otherwise
    )
    # Remove frames where bacterial area is x times average area
    if settings['exclude measurement when above x times average area']:
        df['area'] = np.where(
            df['area'] <= (df['average_area'] * settings['exclude measurement when above x times average area']),
            df['area'],  # track is fine
            np.NaN  # delete otherwise
        )
    # set zeroes in area to NaN
    # tracker.py sets width/height as 0 if it can't connect tracks,
    # thus every data point with area == 0 is suspect
    df.loc[df['area'] == 0, 'area'] = np.NaN

    # remove too short frames
    df['length'] = (df.groupby('TRACK_ID')['POSITION_T'].transform('last') -
                    df.groupby('TRACK_ID')['POSITION_T'].transform('first') + 1
                    ).astype(np.uint16)
    df['area'] = np.where(
        df['length'] >= minimal_length_frames,
        df['area'],  # track is fine
        np.NaN  # delete otherwise
    )

    # remove all rows with a NaN in them - this gets rid of empty/short tracks and empty/suspect measurements
    # As we'll later need only the remaining areas, we'll drop the NaNs
    if settings['verbose']:
        logger.debug('Dropping NaN values from df')
    df.dropna(inplace=True, subset=['area'])

    # reset index to calculate track_change again
    if settings['verbose']:
        logger.debug('Re-indexing')
    df.reset_index(drop=True, inplace=True)
    if df.shape[0] < minimal_length_frames:
        logger.warning(
            'File is empty/of insufficient length after initial clean-up. '
            'Minimal size: {}, length: {}, path: {}'.format(
                minimal_length_frames, df.shape[0], path_to_file
            )
        )
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
            settings['percent quantiles excluded area'],
            (1 - settings['percent quantiles excluded area'])
        ])
        logger.info('Area quartiles: 10%: {:.2f}, 90%: {:.2f}'.format(
            q1_area, q3_area, ))
    else:  # get everything
        q1_area = -1
        q3_area = np.inf
    if settings['try to omit motility outliers']:
        df['distance'] = np.sqrt(np.square(df['POSITION_X'].diff()) +
                                 np.square(df['POSITION_Y'].diff())) / df['POSITION_T'].diff()
        df.loc[track_start, ['distance']] = 0
        # @todo: change to groupby
        q1_dist, q3_dist = df['distance'].quantile(q=[0.25, 0.75])  # IQR
        distance_outlier = (q3_dist - q1_dist) * 3 + q3_dist  # outer fence
        df['distance'] = np.where(df['distance'] > distance_outlier, 1, 0).astype(np.int8)
        distance_outlier_percents = df['distance'].sum() / df.shape[0]
        logger.info('25/75 % Distance quartiles: {:.3f}, {:.3f} upper outliers: {:.3f} '
                    'counts: {}, of all entries: {:.4%}'
                    ''.format(q1_dist, q3_dist, distance_outlier, df['distance'].sum(), distance_outlier_percents))

        if distance_outlier_percents > settings['stop excluding motility outliers if total count above percent']:
            logger.warning(
                'Motility outliers more than {:.2%} of all data points ({:.2%}); recommend to '
                're-analyse file with outlier removal changed if upper quartile is especially low'
                '(Quartile: {:.3f})'.format(
                    settings['stop excluding motility outliers if total count above percent'],
                    distance_outlier_percents,
                    q3_dist
                )
            )
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
        good_track_result, kick_reason = find_good_tracks(
            df_passed=df,
            lower_boundary=q1_area,
            upper_boundary=q3_area,
            start=start,
            stop=stop,
            settings=settings,
            frame_height=frame_height,
            frame_width=frame_width,
            minimal_length_frames=minimal_length_frames,
        )
        kick_reasons[kick_reason] += 1
        # if good_track_result is empty, skip rest:
        if not good_track_result:
            continue
        # get longest track from good_track_result:
        good_selection = 0  # @todo: allow switch between longest/first fragment
        if len(good_track_result) > 1:
            good_comparator = 0
            for idx_good, (good_start, good_stop) in enumerate(good_track_result):
                curr_length = good_stop - good_start + 1
                if curr_length > good_comparator:
                    good_selection = idx_good
                    good_comparator = curr_length
        good_start, good_stop = good_track_result[good_selection]
        # limit track length
        if limit_track_to_frames:  # 0 == False
            # Set limit to start time + limit
            limit_track_length_curr = limit_track_to_frames + df.loc[good_start, 'POSITION_T'] - 1
            # get index of time point closest to limit or maximum
            if not settings['limit track length exactly']:
                good_stop_curr = df.loc[good_start:good_stop, 'POSITION_T'].where(
                    df.loc[good_start:good_stop, 'POSITION_T'] <= limit_track_length_curr).idxmax()
            else:
                good_stop_curr = df.loc[good_start:good_stop, 'POSITION_T'].where(
                    df.loc[good_start:good_stop, 'POSITION_T'] == limit_track_length_curr).idxmax()
            if np.isnan(good_stop_curr):
                continue
            good_stop = good_stop_curr
            # Exclude NaNs in case no index can be found within returned track
        good_track.append((good_start, good_stop))
    logger.info('All tracks before fine selection: {}, left over: {}, difference: {}'.format(
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
    kick_reasons_string = 'Total: {8}; size < 600: {7}; holes > 6: {6}; ' \
                          'distance outlier: {5}; duration 5% over size: {4}; ' \
                          'area out of bounds: {3}; ratio wrong: {2}; ' \
                          'screen edge: {1}; passed: {0}'.format(*kick_reasons, sum(kick_reasons))
    if kick_reasons[0] < 1000 and kick_reasons[0] / sum(kick_reasons) < 0.3:
        logger.warning('Low amount of accepted tracks')
        logger.warning(kick_reasons_string)
    else:
        logger.info(kick_reasons_string)

    if not good_track:  # If we are left with no tracks, we can stop here
        end_string = 'File {} has no acceptable tracks.'.format(path_to_file)
        logger.warning(end_string)
        return None

    # Convert good_track to list
    # (a lot faster than setting slices of np.array to true for some reason)
    df['good_track'] = np.zeros(df.shape[0], dtype=np.int8)
    set_good_track_to_true = []
    for (start, stop) in good_track:
        set_good_track_to_true.extend(range(start, (stop + 1), 1))
    df.loc[set_good_track_to_true, ['good_track']] = 1
    del set_good_track_to_true

    # Reset df to important parts
    if settings['verbose']:
        logger.debug('Resetting df')
    df_passed_columns = ['TRACK_ID', 'POSITION_T', 'POSITION_X', 'POSITION_Y', 'WIDTH', 'HEIGHT', 'DEGREES_ANGLE']
    df = df.loc[df['good_track'] == 1, df_passed_columns]
    df.reset_index(inplace=True)
    save_path = os.path.join(results_directory, file_name) + '_{}{}'
    if settings['store processed .csv file']:
        save_df_to_csv(df=df, save_path=save_path.format('selected_data', '.csv'))
    return df


def evaluate_tracks(path_to_file, results_directory, df=None, settings=None, fps=None, **_):
    """
    calculate additional info from provided .csv/data frame

    :param path_to_file: .csv file
    :param results_directory: path to results directory
    :param df: optional pandas data frame
    :param settings: tracking.ini settings
    :type settings: dict
    :param fps: frame per second value
    :type fps: float
    :return: modified df, statistics as pandas data frame
    """
    logger = logging.getLogger('ysmr').getChild(__name__)
    settings = get_configs(settings)  # Get settings
    if settings is None:
        logger.critical('No settings provided.')
        return None
    # Set up and check some basic stuff
    if fps is None or fps <= 0 or settings['force tracking.ini fps settings']:
        if settings['frames per second'] > 0:
            fps = settings['frames per second']
        else:
            logger.critical('fps value is negative or zero; cannot continue.')
            return None
    file_name = os.path.basename(path_to_file)
    file_name = os.path.splitext(file_name)[0]
    if type(df) is not pd.core.frame.DataFrame:  # In case we didn't get a data frame
        if settings['verbose']:
            logger.debug('Handing string to get_data {}'.format(path_to_file))
        df = get_data(path_to_file)
    if df is None:  # get_data() returns None in case of errors
        logger.critical('Error reading data frame from file {}'.format(path_to_file))
        return None
    diff_tracks_start, track_change = different_tracks(df)
    px_to_micrometre = settings['pixel per micrometre']
    # @todo: allow user customisation of plot title name?
    # Set up plot title name
    plot_title_name = file_name.replace('_', ' ')
    if '_selected_data' in file_name:
        plot_title_name = plot_title_name[:-len('_selected_data')]
    # plot_title_name = plot_title_name.split(sep='.', maxsplit=1)[0]
    original_plot_date = plot_title_name[:12]
    # Add time/date to title of plot
    if original_plot_date.isdigit() and len(original_plot_date) == 12:
        try:
            original_plot_date = strftime('%d. %m. \'%y,', strptime(str(original_plot_date), '%y%m%d%H%M%S'))
            plot_title_name = '{} {}'.format(original_plot_date, plot_title_name[12:])
        except ValueError:
            pass

    # Format general save path
    save_path = os.path.join(results_directory, file_name) + '_{}{}'

    # set up some overall values
    if settings['verbose']:
        logger.debug('Calculating x_delta, y_delta, t_delta, travelled_dist')
    df['x_delta'] = df['POSITION_X'].diff()
    df['y_delta'] = df['POSITION_Y'].diff()
    df['t_delta'] = df['POSITION_T'].diff()
    # Set correct values for track starts
    df.loc[diff_tracks_start, ['x_delta', 'y_delta']] = 0
    df.loc[diff_tracks_start, ['t_delta']] = 1
    for letter in ['x', 'y', 't']:  # validate
        item = '{}_delta'.format(letter)
        if df[item].isnull().any():  # check if any value is still NaN
            logger.critical('{} has NaN value(s) after '
                            'clean-up at position(s): {}'.format(item, np.where(df[item].isnull())[0]))
            logger.critical('{} track starts: {}'.format(item, diff_tracks_start))

    df['t_norm'] = df['POSITION_T'].sub(
        df.groupby('TRACK_ID')['POSITION_T'].transform('first')).astype(np.int32)
    if any(df['t_norm'] < 0):  # validate
        logger.critical('POSITION_T contains negative values')
        return None

    df['WIDTH'] = df['WIDTH'] / px_to_micrometre
    df['HEIGHT'] = df['HEIGHT'] / px_to_micrometre
    df['area'] = df['WIDTH'] * df['HEIGHT']  # calculate area of bacteria in micrometre**2
    df['bac_length'] = np.where(df['WIDTH'] >= df['HEIGHT'], df['WIDTH'], df['HEIGHT']).astype(np.float16)

    if settings['verbose']:
        logger.debug('Starting with statistical calculations per track')

    # travelled_distance = square root(delta_x^2 + delta_y^2) / px to micrometre ratio
    df['travelled_dist'] = np.sqrt(np.square(df['x_delta']) + np.square(df['y_delta'])) / px_to_micrometre
    df['moving'] = df['travelled_dist'] / df['t_delta']
    # get rid of rounding errors, convert to binary:
    df['moving'] = np.where(df['moving'] > 10 ** -3, 1, 0).astype(np.int8)
    if int(round(fps, 0)) & 1 == 0:  # if fps is even
        max_kernel = int(round(fps, 0)) + 1
    else:
        max_kernel = int(round(fps, 0))

    # median filter the values to spot general null points in movement
    for kernel_size in [3, max_kernel]:
        df['moving'] = df.groupby('TRACK_ID')['moving'].transform(medfilt, kernel_size=kernel_size)

    angle_diff = settings['compare angle between n frames']
    x_diff_track_for_angle = df.groupby('TRACK_ID')['POSITION_X'].diff(angle_diff)  # .fillna(method='bfill')
    y_diff_track_for_angle = df.groupby('TRACK_ID')['POSITION_Y'].diff(angle_diff)  # .fillna(method='bfill')
    df['angle_diff'] = np.arctan2(x_diff_track_for_angle, y_diff_track_for_angle)  # rad

    # Angle distribution histogram
    if settings['save angle distribution plot / bins']:  # 0 == False
        # takes angle_diff as rad
        angle_distribution_plot(df=df,
                                bins_number=settings['save angle distribution plot / bins'],
                                plot_title_name=plot_title_name,
                                save_path=save_path.format('angle_histogram', '.png')
                                )
    min_angle = settings['minimal angle in degrees for turning point']
    df['angle_diff'] = np.degrees(df['angle_diff'])  # deg
    # Convert to angle difference between 0 and 180
    df['angle_diff'] = abs(df.groupby('TRACK_ID')['angle_diff'].diff().fillna(0))
    df['angle_diff'] = np.where(360 - df['angle_diff'] <= df['angle_diff'],
                                360 - df['angle_diff'],
                                df['angle_diff']
                                ).astype(np.int32)
    df['turn_points'] = np.where(
        (df['angle_diff'] > min_angle) & (df['moving'] == 1),
        df['angle_diff'],
        0).astype(np.int32)

    # normalise x/y coordinates, convert from px to micrometre
    df['x_norm'] = (df['POSITION_X'].sub(df.groupby('TRACK_ID')['POSITION_X'].transform('first'))) / px_to_micrometre
    df['y_norm'] = (df['POSITION_Y'].sub(df.groupby('TRACK_ID')['POSITION_Y'].transform('first'))) / px_to_micrometre

    # get local maxima
    df['turn_points'] = df.groupby('TRACK_ID')['turn_points'].apply(argrelextrema_groupby)
    # Convert to binary
    df['turn_points'] = np.where(df['turn_points'] == 0, 0, 1).astype(np.int8)
    # start counts as TP so we can later group by TPs
    df.loc[diff_tracks_start, ['turn_points']] = 1
    # get indices of turning points
    tp_start, _ = different_tracks(df, column='turn_points')
    # as different_tracks returns changes from 1 to 0 and 0 to 1, we need only every other entry
    tp_start = tp_start[::2]
    tp_start.append(df.index.max())  # last item added as stop
    df['bac_average_size'] = df.groupby('TRACK_ID')['bac_length'].transform('mean')
    # Create one column in which each TP gets a unique ID
    df['tp_of_tracks'] = np.zeros(df.shape[0], dtype=np.uint64)
    for i, (start, stop) in enumerate(zip(tp_start[:-1], tp_start[1:])):
        df.loc[start:stop - 1, ['tp_of_tracks']] = i
    df['tp_of_tracks'] = np.where(df['moving'] == 0, np.nan, df['tp_of_tracks'])
    df['tp_dist'] = df.groupby('tp_of_tracks')['travelled_dist'].transform('sum')
    # limit to 10 or half minimal length in seconds/limit track length to x seconds, whichever is shortest
    # halves are used as some tracks might be shorter when limit exactly is not in use
    seconds_difference_list = [10]
    if 0 < settings['minimal length in seconds'] / 2 < 10:
        seconds_difference_list.append(settings['minimal length in seconds'] / 2)
    if 0 < settings['limit track length to x seconds'] / 2 < 10:
        seconds_difference_list.append(settings['limit track length to x seconds'] / 2)
    seconds_difference = min(seconds_difference_list)
    # Get largest displacement per bacterium divided by individual length over ~10 s
    df['x_fps_diff'] = df.groupby('TRACK_ID')['x_norm'].diff(int(round((fps * seconds_difference), 0)))
    df['y_fps_diff'] = df.groupby('TRACK_ID')['y_norm'].diff(int(round((fps * seconds_difference), 0)))
    df['pdist_series_max'] = np.sqrt(np.square(df['x_fps_diff']) + np.square(df['y_fps_diff']))
    df['pdist_series_max'] = df.groupby('TRACK_ID')['pdist_series_max'].transform('max')
    # Divide by bac. length
    df['pdist_series_max'] = df['pdist_series_max'] / df['bac_average_size']
    # longest
    df['tp_dist_by_size_max'] = df.groupby('TRACK_ID')['tp_dist'].transform('max') / df['bac_average_size']

    # Phenotypes: 2: motile; 1: twitching; 0: immotile
    df['motility_phenotype'] = np.zeros(df.shape[0], dtype=np.int8)
    df['motility_phenotype'] = np.where(
        ((df['pdist_series_max'] > 1.5) & (df['tp_dist_by_size_max'] > 5)),  # motile
        2,
        df['motility_phenotype']
    )
    df['motility_phenotype'] = np.where(
        ((df['pdist_series_max'] > 1.5) & (df['tp_dist_by_size_max'] <= 5)),  # twitching
        1,
        df['motility_phenotype']
    )

    motility_categories = ['immotile', 'twitching', 'motile']
    motility_categories = [i for i in range(len(motility_categories))]

    # df['motility_phenotype'].replace(  # replace 0 / 1 / 2 with immotile / twitching / motile
    #     {value: key for key, value in zip(motility_categories, range(0, len(motility_categories) + 1))},
    #     inplace=True)

    # Source: ALollz, stackoverflow.com/questions/51064346/
    # Get largest displacement during track
    pdist_series = df.groupby('TRACK_ID').apply(lambda l: dist.pdist(np.array(list(zip(l.x_norm, l.y_norm)))).max())
    # lambda l: np.nanmax(
    #             dist.pdist(
    #                 np.array(list(zip(l.x_norm, l.y_norm)))
    #             )
    #         )
    time_series = df.groupby('TRACK_ID')['t_norm'].agg('last')
    motile_total_series = df.groupby('TRACK_ID')['moving'].agg('sum')
    motile_series = motile_total_series / (time_series + 1) * 100  # off-by-one error
    time_series = (time_series + 1) / fps  # off-by-one error
    dist_series = df.groupby('TRACK_ID')['travelled_dist'].agg('sum')
    acr_series = np.sqrt(
        np.square(df.groupby('TRACK_ID')['x_norm'].agg('last')) +
        np.square(df.groupby('TRACK_ID')['y_norm'].agg('last'))
    )
    speed_series = pd.Series(
        (np.where(motile_total_series != 0,
                  dist_series / time_series,
                  0)), index=time_series.index)
    # Avoid div. by 0:
    acr_series = pd.Series(
        (np.where(dist_series != 0,
                  acr_series / dist_series,
                  0
                  )), index=time_series.index)

    # Remove truning points from immotile tracks
    df['turn_points'] = np.where(
        df['motility_phenotype'] != 0,  # If tracks are not immotile
        df['turn_points'],  # Keep as-is
        0  # No TP otherwise
    )

    # Set start of each track as a turning point
    # This way we can group by turning points afterwards
    df.loc[diff_tracks_start, ['turn_points']] = 1

    # Subtract one as each track starts with a TP and multiply by fps;
    # As we'll divide by time afterwards so we can get 1/s if there is one positive in one second
    turn_per_s_series = (df.groupby('TRACK_ID')['turn_points'].agg('sum') - 1) * fps
    turn_per_s_series = pd.Series(
        (np.where(motile_total_series != 0,
                  turn_per_s_series / motile_total_series,
                  0)), index=time_series.index)

    bac_length_series = pd.Series(df.groupby('TRACK_ID')['bac_length'].agg('mean'))
    displ_bac_series = pd.Series(
        (np.where(
            bac_length_series != 0,
            pdist_series / bac_length_series,
            0
        )), index=time_series.index)
    track_id = df.groupby('TRACK_ID')['TRACK_ID'].agg('last')
    mot_phenotype = df.groupby('TRACK_ID')['motility_phenotype'].agg('last')

    name_of_columns = [
        'Turn Points (TP/s)',  # 0
        'Distance (µm)',  # 1
        'Speed (µm/s)',  # 2
        'Time (s)',  # 3
        'Displacement (µm)',  # 4
        'Perc. Motile',  # 5
        'Arc-Chord Ratio',  # 6
        'Bacteria Length',  # 7
        'Displacement divided by length',  # 8
        'Motility Phenotype',  # 9
        'TRACK_ID',  # 10
    ]
    # Create df for statistics
    df_stats = pd.concat(
        [turn_per_s_series,  # 0
         dist_series,  # 1
         speed_series,  # 2
         time_series,  # 3
         pdist_series,  # 4
         motile_series,  # 5
         acr_series,  # 6
         bac_length_series,  # 7
         displ_bac_series,  # 8
         mot_phenotype,  # 9
         track_id,  # 10
         ],
        keys=name_of_columns, axis=1
    )
    del turn_per_s_series, dist_series, speed_series, time_series, pdist_series, motile_series

    if settings['store generated statistical .csv file']:
        # df_stats_columns = name_of_columns
        # switch IDs to first column
        # df_stats_columns[0], df_stats_columns[-1] = df_stats_columns[-1], df_stats_columns[0]  # TRACK_ID first
        save_df_to_csv(
            df=df_stats,  # .reindex(columns=df_stats_columns),
            save_path=save_path.format('statistics', '.csv')
        )
        # df_stats.reindex(columns=name_of_columns)
    # OH GREAT MOTILITY ORACLE, WHAT WILL MY BACTERIAS MOVES BE LIKE?
    nonmotile = df['motility_phenotype'].where(
        df['motility_phenotype'] == motility_categories[0]).count() / df.shape[0]
    twitching = df['motility_phenotype'].where(
        df['motility_phenotype'] == motility_categories[1]).count() / df.shape[0]
    motile = df['motility_phenotype'].where(
        df['motility_phenotype'] == motility_categories[2]).count() / df.shape[0]
    logger.info('Nonmotile: {:.2%}, twitching: {:.2%}, motile: {:.2%}'.format(nonmotile, twitching, motile))
    q1_time, q2_time, q3_time = np.quantile(df_stats[name_of_columns[3]], (0.25, 0.5, 0.75))  # 'Time (s)',  # 3
    logger.debug('Time duration of selected tracks min: {:.3f}, max: '
                 '{:.3f}, Quantiles (25/50/75%): {:.3f}, {:.3f}, {:.3f}'
                 ''.format(min(df_stats[name_of_columns[3]]), max(df_stats[name_of_columns[3]]),
                           q1_time, q2_time, q3_time))

    # Prepare cut off list
    cut_off = settings['split results by (Turn Points / Distance / Speed / Time / Displacement / perc. motile)']
    cut_off_parameter = None
    for name in name_of_columns:
        if cut_off.lower() in name.lower():
            cut_off_parameter = name
            break
    if not cut_off_parameter:
        logger.warning(
            'Setting \'split results by parameter (Turn Points / Distance / Speed / Time / Displacement / % motile)\' '
            'could not be assigned, reverted to \'perc. motile\'.')
        cut_off_parameter = name_of_columns[5]

    cut_off_list = settings['split violin plots on']

    if cut_off_parameter == name_of_columns[9]:
        cut_off_list = motility_categories

    name_all_categories = 'All'
    if cut_off_parameter == name_of_columns[5]:
        cut_off_precursor = [(a, b, '{:.1f}% - {:.1f}%'.format(a, b)) for a, b in
                             zip(cut_off_list[:-1], cut_off_list[1:])]
    elif cut_off_parameter == name_of_columns[9]:
        cut_off_precursor = [(0, 0.001, 'Immotile'), (1, 1.001, 'Twitching'), (2, 2.001, 'Motile'), ]
    else:
        cut_off_precursor = [(a, b, '{:.2f} - {:.2f}'.format(a, b)) for a, b in
                             zip(cut_off_list[:-1], cut_off_list[1:])]
    cut_off_list = [(np.NINF, np.inf, name_all_categories)]  # So one x contains all values
    cut_off_list.extend(cut_off_precursor)

    # Calculate df_stats_seaborne for statistical plots
    cut_off_category = 'Categories ({})'.format(cut_off_parameter)
    df_stats[cut_off_category] = name_all_categories
    df_stats_seaborne = df_stats.copy()

    # To drop unassigned values later
    df_stats_seaborne[cut_off_category] = np.NaN
    for index_cut_off, (low, high, category) in enumerate(cut_off_list):
        # Since we'll concatenate df_stats to df_stats_seaborne, which already contains all
        if category == name_all_categories:
            continue
        # inclusive low, exclusive high
        df_stats_seaborne[cut_off_category] = np.where(
            (low <= df_stats[cut_off_parameter]) &
            (high > df_stats[cut_off_parameter]),
            index_cut_off,
            df_stats_seaborne[cut_off_category]
        )
    # All np.NaN in column get replaced with str when str values are set within column -> easier to exchange later
    df_stats_seaborne.dropna(subset=[cut_off_category], inplace=True)

    # Exchange int values in 'Categories' for correct labels
    df_stats_seaborne[cut_off_category].replace(  # name_all_categories is not present and can be skipped
        {value: key for key, value in zip([i for (_, _, i) in cut_off_list[1:]], range(1, len(cut_off_list)))},
        inplace=True)

    # Put name_all_categories and assigned categories in one df
    df_stats_seaborne = pd.concat([df_stats, df_stats_seaborne], ignore_index=True)

    # Get name of categories, assign values in ascending order, sort df_stats_seaborne[cut_off_category] by order/dict
    categories = {key: value for key, value in zip([i for (_, _, i) in cut_off_list], range(0, len(cut_off_list)))}
    # sort df_stats_seaborne by generated dict key:value pairs (in order of cut_off_list)
    df_stats_seaborne = df_stats_seaborne.iloc[df_stats_seaborne[cut_off_category].map(categories).sort_values().index]

    if settings['save large plots'] or settings['save rose plot']:
        # distance colour for plots
        distance_min = df_stats[name_of_columns[1]].min()  # 'Distance (micrometre)',  # 1
        distance_max = df_stats[name_of_columns[1]].max()
        df['distance_colour'] = df.groupby('TRACK_ID')['travelled_dist'].transform('sum') - distance_min
        df['distance_colour'] = df['distance_colour'] / df['distance_colour'].max()

        if settings['save large plots']:
            large_xy_plot(df=df,
                          plot_title_name=plot_title_name,
                          save_path=save_path.format('Bac_Run_Overview', '.png'),
                          dist_min=distance_min,
                          dist_max=distance_max,
                          px_to_micrometre=px_to_micrometre,
                          )
        if settings['save rose plot']:
            rose_graph(df=df,
                       plot_title_name=plot_title_name,
                       save_path=save_path.format('rose_graph', '.png'),
                       dist_min=distance_min,
                       dist_max=distance_max)
    violin_plots = []
    if settings['save turning point violin plot']:
        violin_plots.append((
            name_of_columns[0], 'turning_points',
            settings['turning point violin plot min'],
            settings['turning point violin plot max'],
        ))

    if settings['save length violin plot']:
        violin_plots.append((
            name_of_columns[1], 'distance',
            settings['length violin plot min'],
            settings['length violin plot max'],
        ))

    if settings['save speed violin plot']:
        violin_plots.append((
            name_of_columns[2], 'speed',
            settings['speed violin plot min'],
            settings['speed violin plot max'],
        ))

    if settings['save time violin plot']:
        violin_plots.append((
            name_of_columns[3], 'time_plot',
            settings['time violin plot min'],
            settings['time violin plot max'],
        ))
    if settings['save displacement violin plot']:
        violin_plots.append((
            name_of_columns[4], 'displacement',
            settings['displacement violin plot min'],
            settings['displacement violin plot max'],
        ))

    if settings['save percent motile plot']:
        violin_plots.append((
            name_of_columns[5], 'perc_motile',
            settings['percent motile plot min'],
            settings['percent motile plot max'],
        ))

    if settings['save acr violin plot']:
        violin_plots.append((
            name_of_columns[6], 'arc-chord_ratio',
            settings['acr violin plot min'],
            settings['acr violin plot max'],
        ))

    for category, plot_name, y_min, y_max in violin_plots:
        violin_plot(
            df=df_stats_seaborne,
            save_path=save_path.format(plot_name, '.png'),
            cut_off_category=cut_off_category,
            category=category,
            cut_off_list=cut_off_list,
            verbose=settings['verbose'],
            y_min=y_min,
            y_max=y_max,
            plot_title_name=plot_title_name,
        )

    df_passed_columns = [
        'TRACK_ID', 'POSITION_T', 'POSITION_X', 'POSITION_Y', 'WIDTH', 'HEIGHT', 'DEGREES_ANGLE',
        'angle_diff', 'moving', 'turn_points', 'tp_of_tracks', 'travelled_dist', 'motility_phenotype'
    ]
    df = df.loc[:, df_passed_columns]  # drop unnecessary columns
    if settings['store final analysed .csv file']:
        save_df_to_csv(
            df=df,
            save_path=save_path.format('analysed', '.csv')
        )

    end_string = 'Done evaluating file {}'.format(file_name)
    logging.info(end_string)
    return df, df_stats


def annotate_video(video_path, df, output_save=True, settings=None, result_folder=None, select_subtype=None, **_):
    """
    Annotate video with positions and properties from data frame

    :param video_path: path to video file
    :param df: pandas data frame
    :param output_save: Wheter to display or save the output
    :type output_save: bool
    :param settings: tracking.ini settings
    :param result_folder: Path to folder in which to save the video file.
        If None is given, will use video base folder.
    :param select_subtype: Whether to only display a subtype as set in '
        motility_phenotype' column (immotile, twitching, motile).
    :return: None
    """
    logger = logging.getLogger('ysmr').getChild(__name__)
    settings = get_configs(settings)  # Get settings
    # We may have to set the log level/loggers again due to multiprocessing
    get_loggers(
        log_level=settings['log_level'],
        logfile_name=settings['log file path'],
        short_stream_output=settings['shorten displayed logging output'],
        short_file_output=settings['shorten logfile logging output'],
        log_to_file=settings['log to file'])
    try:
        cap = cv2.VideoCapture(video_path)
    except (IOError, OSError) as io_error:
        logger.exception('Cannot open file {} due to error: {}'.format(video_path, io_error))
        return None

    if not result_folder:
        result_folder = create_results_folder(video_path)

    if type(df) is not pd.core.frame.DataFrame:  # In case we didn't get a data frame
        if settings['verbose']:
            logger.debug('Handing string to get_data {}'.format(df))
        dtype = {
            'TRACK_ID': np.int64,
            'POSITION_T': np.int64,
            'POSITION_X': np.float64,
            'POSITION_Y': np.float64,
            'motility_phenotype': object,
            'moving': np.int8,
            'turn_points': np.int8,

        }
        df = get_data(df, dtype=dtype)

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    (frame_height, frame_width) = (int(cap.get(4)), int(cap.get(3)))  # Image dimensions
    curr_frame = 0
    try:
        fps_of_file = cap.get(cv2.CAP_PROP_FPS)
    except Exception as ex:
        template = 'An exception of type {0} occurred while accessing fps from file {2}. Arguments:\n{1!r}'
        logger.exception(template.format(type(ex).__name__, ex.args, video_path))
        if settings['frames per second'] <= 0:
            logger.critical('User defined fps unacceptable: type: {} value: {}'.format(
                type(settings['frames per second']), settings['frames per second']))
            return None
        else:
            fps_of_file = settings['frames per second']
    pathname, filename_ext = os.path.split(video_path)
    filename = os.path.splitext(filename_ext)[0]
    if select_subtype is None:
        output_video_name = os.path.join(result_folder, '{}_annotated_output{}'.format(
            filename, settings['save video file extension']))
    else:
        if isinstance(select_subtype, int):
            subtype = ['immotile', 'twitching', 'motile']
            select_subtype = subtype[select_subtype]
        output_video_name = os.path.join(result_folder, '{}_subtype_{}_annotated_output{}'.format(
            select_subtype, filename, settings['save video file extension']))

    _, filename_ext = os.path.split(video_path)
    filename = os.path.splitext(filename_ext)[0]
    if output_save:
        out = cv2.VideoWriter(
            output_video_name,
            cv2.VideoWriter_fourcc(*'{}'.format(settings['save video fourcc codec'])),  # Codec *'MJPG'  # X264  # H264
            fps_of_file,  # FPS
            (frame_width, frame_height)  # Dimensions
        )
    while True:  # Loop over video
        if not output_save:
            # slow down, otherwise tends to be too fast
            sleep(1 / fps_of_file)
        ret, frame = cap.read()
        if not ret and (frame_count == curr_frame + 1 or  # some file formats skip one frame
                        frame_count == curr_frame) and frame_count >= settings['minimal frame count']:
            # If a frame could not be retrieved and the minimum frame nr. has been reached
            logger.debug('Frames from file {} read.'.format(filename_ext))
            break
        elif not ret:  # Something must've happened, user decides if to proceed
            logger.critical('Error during cap.read() with file {}'.format(video_path))
            break
        if select_subtype is None:
            curr_boxes = df[df['POSITION_T'] == curr_frame]
        else:
            curr_boxes = df[(df['POSITION_T'] == curr_frame) & (df['motility_phenotype'] == select_subtype)]
        for _, row in curr_boxes.iterrows():
            curr_color = (0, 255, 0)
            size_circle = 0
            line_thickness_text = 0
            if row['moving'] == 0:
                curr_color = (15, 165, 253)
            elif row['turn_points'] == 1:
                curr_color = (255, 255, 255)
                size_circle = 1
                line_thickness_text = 1
            text = '{}'.format(int(row['TRACK_ID']))
            # Display object ID:
            cv2.putText(frame,
                        text,
                        (int(row['POSITION_X']) - 10, int(row['POSITION_Y']) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.3,
                        curr_color,
                        line_thickness_text)
            # Display centroid:
            cv2.circle(frame,
                       (int(row['POSITION_X']), int(row['POSITION_Y'])),
                       size_circle,
                       curr_color,
                       -1)
        if output_save:
            out.write(frame)
        else:
            cv2.putText(frame,  # image
                        '{:>6}'.format(curr_frame),  # text
                        (20, 20),  # xy coordinates
                        cv2.FONT_HERSHEY_SIMPLEX,  # font
                        0.7,  # text size
                        (220, 220, 60),  # colour
                        1  # line thickness
                        )
            if select_subtype is None:
                cv2.imshow('{}'.format(filename), frame)
            else:
                cv2.imshow('{} {}'.format(filename, select_subtype), frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):  # Interrupt display on 'q'-keypress
                logger.error('Processing file interrupted by user: {}'.format(video_path))
                break
        curr_frame += 1
    if output_save:
        out.release()
        logger.debug('Output video file: {}'.format(output_video_name))
    else:
        cv2.destroyAllWindows()  # Close active windows
    cap.release()
