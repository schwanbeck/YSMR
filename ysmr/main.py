#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Copyright 2019, 2020 Julian Schwanbeck (schwan@umn.edu)
https://github.com/schwanbeck/YSMR
##Explanation
This file starts YSMR.
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
from datetime import datetime
from time import sleep

from ysmr.helper_file import (check_logfile, collate_results_csv_to_xlsx, create_results_folder, elapsed_time,
                              get_any_paths, get_configs, get_loggers, log_infos, logging_configurer,
                              logging_listener, metadata_file, shutdown, stop_logging_queue)
from ysmr.track_eval import annotate_video, evaluate_tracks, select_tracks, track_bacteria

__all__ = ['analyse', 'ysmr']


def analyse(path, settings=None, result_folder=None, return_df=False, **kwargs):
    """Starts analysis function (evaluate_tracks, select_tracks, track_bacteria),
    depending on received file type / -ending.
    Continues with analysis depending on provided settings.

    :param path: path to file
    :param settings: tracking.ini settings
    :type settings: dict
    :param result_folder: folder to save results in
    :param return_df: whether to return resulting data frame
    :type return_df: bool
    :param kwargs: kwargs will be saved to meta data file (_meta.json)
    :return: pandas data frame(s), depending on last analysis step, bool if return_df is False, None if error occurred
    """
    t_one = datetime.now()  # to get rough time estimation
    settings = get_configs(settings)
    if settings is None:
        return None
    get_loggers(
        log_level=settings['log_level'],
        logfile_name=settings['log file path'],
        short_stream_output=settings['shorten displayed logging output'],
        short_file_output=settings['shorten logfile logging output'],
        log_to_file=settings['log to file'],
        settings=settings,
    )
    logger = logging.getLogger('ysmr').getChild(__name__)
    return_value = None
    if result_folder is None:
        result_folder = create_results_folder(path)
    logger.debug('Starting process. PID: {} Result folder: {}'.format(os.getpid(), result_folder))
    # path = os.path.realpath(path)
    # See if we need to evaluate for anything, otherwise we'll skip evaluate_tracks()
    plots_eval = any([
        settings['store generated statistical .csv file'],
        settings['store final analysed .csv file'],
        settings['save large plots'],
        settings['save rose plot'],
        settings['save time violin plot'],
        settings['save acr violin plot'],
        settings['save length violin plot'],
        settings['save turning point violin plot'],
        settings['save speed violin plot'],
        settings['save angle distribution plot / bins'],
        settings['collate results csv to xlsx'],
        settings['save video'],
    ])
    # set values to None
    df, fps, f_height, f_width, csv_file = [None] * 5

    while True:  # so we can break on error
        finished_files = ['_analysed.csv', '_statistics.csv', '_annotated_output.']
        if any(file_ext in path for file_ext in finished_files):  # Already evaluated file
            logger.warning('File already evaluated. File: {}'.format(path))
            return_value = None
            break
        if '.csv' not in path:  # as long as it's not a .csv, it should be a video:
            if settings['verbose']:
                logging.debug('File ends not in .csv, file is assumed to be a video.')
            track_result = track_bacteria(video_path=path, settings=settings, result_folder=result_folder)
            if track_result is None:
                logger.warning('Error during video analysis of file {}.'.format(path))
                return_value = None
                break
            (df, fps, f_height, f_width, csv_file) = track_result
            return_value = df
        # save fps and frame dimensions in metadata json
        meta_data = metadata_file(
            # meta.json file will be searched for in provided folder and parent folder
            path=os.path.join(result_folder, os.path.basename(path)),
            additional_search_paths=path,
            verbose=settings['verbose'],
            fps=fps,
            frame_height=f_height,
            frame_width=f_width,
            **kwargs,
        )
        if settings['debugging']:
            for key, value in meta_data.items():
                logger.debug('{}: {}'.format(key, value))
        # if it's not evaluated yet:
        if 'selected_data.csv' not in path and (plots_eval or settings['store processed .csv file']):
            df = select_tracks(
                path_to_file=path,
                df=df,
                results_directory=result_folder,
                # fps=fps,  # taken from **meta_data
                # frame_height=f_height,
                # frame_width=f_width,
                settings=settings,
                **meta_data
            )
            if df is None:
                logger.warning('Error during video analysis of file {}.'.format(path))
                return_value = None
                break
            return_value = df
        # Check if anything needs evaluation
        if plots_eval:
            return_value = evaluate_tracks(
                path_to_file=path,
                results_directory=result_folder,
                df=df,
                settings=settings,
                # fps=fps,  # taken from **meta_data
                **meta_data
            )
            if settings['save video'] and '.csv' not in path:
                annotate_video(
                    video_path=path,
                    df=return_value[0],
                    settings=settings,
                    result_folder=result_folder
                )
            elif settings['save video'] and '.csv' in path:
                logger.warning(
                    '\'save video\' setting is enabled but .csv file was provided. Video can only be annotated '
                    'when ysmr() is given a video as an argument. Optionally use annotate_video() from '
                    'ysmr.track_eval directly.')
        # if nothing is selected for evaluation, and it's specifically a selected_data.csv, something seems wrong
        elif 'selected_data.csv' in path:
            logger.warning('No evaluation set to True in settings. Did not evaluate {}'.format(path))
        break  # all is well

    if settings['delete .csv file after analysis'] and csv_file:
        try:
            os.remove(csv_file)
        except FileNotFoundError:
            pass
        except Exception as ex:
            template = 'An exception of type {0} occurred. Arguments:\n{1!r}'
            logger.exception(template.format(type(ex).__name__, ex.args))
    if return_value is not None:
        end_string = 'Finished with'
        if not return_df:
            return_value = True
    else:
        end_string = 'Error during'
    logger.info('{} process. PID: {}, elapsed time: {}'.format(
        end_string, os.getpid(), elapsed_time(t_one)))
    return return_value


def ysmr(paths=None, settings=None, result_folder=None, multiprocess=False):
    """
    Starts asynchronous multiprocessing of provided video file(s) with analyse().
    Due to the use of multiprocessing, ysmr() should be called in a
    "if __name__ == "__main__":" block. Doing otherwise can lead to
    unexpected behaviour.

    :param settings: tracking.ini settings
    :type settings: dict, str, os.PathLike
    :param paths: path or iterable with paths
    :type paths: str, list, os.PathLike
    :param result_folder: path to result folder, if not provided, one will be created in first path folder
    :type result_folder: str, list, os.PathLike
    :param multiprocess: Whether to run as multiprocess or not. Requires to be run in main block or forked process
     - may lead to unexpected behaviour otherwise.
    :type multiprocess: bool
    :return: list of (finished path, results)
    :rtype paths_finished: list
    """
    t_one = datetime.now()  # to get rough time estimation
    settings = get_configs(settings)  # Get settings
    if settings is None:
        print('Fatal error in retrieving tracking.ini')
    paths_failed = []
    paths_finished = []
    if isinstance(paths, str) or isinstance(paths, os.PathLike):
        paths = [paths]  # convert to list, otherwise for path in paths iterates over characters in string

    settings['log file path'] = check_logfile(path=settings['log file path'])

    if not settings['debugging']:
        settings['logging_queue'] = mp.Manager().Queue(-1)
        listener = mp.Process(target=logging_listener, args=(settings,))
        listener.start()
        logging_configurer(settings)
    else:
        listener = None

    get_loggers(
        log_level=settings['log_level'],
        logfile_name=settings['log file path'],
        short_stream_output=settings['shorten displayed logging output'],
        short_file_output=settings['shorten logfile logging output'],
        log_to_file=settings['log to file'],
        settings=settings
    )
    logger = logging.getLogger('ysmr').getChild(__name__)
    filler_for_logger = log_infos(settings=settings)

    if settings['debugging']:  # multiprocess can be uncommunicative with errors
        result_folder = create_results_folder(path=settings['path to test video'])
        if paths is None:
            path = os.path.expanduser(settings['path to test video'])
        else:
            path = paths[0]
        if not os.path.isfile(path):
            logger.critical('Path to test video may not exist, attempting anyway: {}'.format(path))
        else:
            logger.info('Path: {}'.format(path))
        return analyse(
            path=path,
            settings=settings,
            result_folder=result_folder
        )

    else:
        if settings['select files']:
            if not paths:
                paths = get_any_paths(rename=True, settings=settings)
            if not paths:
                logger.critical('No files selected.')
                stop_logging_queue(logger, settings)
                listener.join()
                return None
        else:
            if not paths:
                paths = [settings['path to test video']]
            logger.info('Test video path selected')
        # Expand user if necessary
        paths_expanded = [os.path.expanduser(path) for path in paths]
        paths = paths_expanded
        for path in paths:
            logger.debug(path)
        logger.info('Total number of files: {}'.format(len(paths)))

        while settings['user input']:  # give user chance to check input
            logger.debug('Waiting for user input.')
            sleep(.1)  # So the logger doesn't interfere with user input
            event = input('Continue? (Y/N): ')
            if 0 < len(event) < 4:
                if event[0].lower() == 'n':
                    logger.info('Process aborted.\n')
                    stop_logging_queue(logger, settings)
                    listener.join()
                    return None
                elif event[0].lower() == 'y':
                    logger.debug('User agreed.')
                    break
        results = {}
        if result_folder is None:
            result_folder = create_results_folder(paths[0])
        if not os.path.isdir(result_folder):
            os.makedirs(result_folder, exist_ok=True)

        # get a pool of worker processes per available core
        if multiprocess:
            # mp.set_start_method('spawn')
            pool = mp.Pool(maxtasksperchild=1)
            for path in paths:
                # Asynchronous calls to track_bacteria() with each path
                results[path] = pool.apply_async(analyse, args=(path, settings, result_folder))
            pool.close()
            pool.join()
        else:
            for path in paths:
                results[path] = analyse(path=path, settings=settings, result_folder=result_folder)
        for path, item in results.items():
            try:
                if multiprocess:
                    result = item.get()
                else:
                    result = item
                if result is None:
                    paths_failed.append(path)
                    paths_finished.append((path, None))
                else:
                    paths_finished.append((path, item))
            except (FileNotFoundError, PermissionError,):
                logger.critical('The file could not be found or opened: {}'.format(path))
            except Exception as exc:
                logger.critical('An exception of type {0} occurred with path {1}. Arguments:'.format(
                    type(exc).__name__, path))
                for line in str(exc.args).splitlines():
                    logger.critical('{}'.format(line))
                logger.exception(exc)
                paths_failed.append(path)
                paths_finished.append((path, None))
                continue
        if paths_failed:
            logger.critical('Failed to analyse {} of {} file(s):'.format(len(paths_failed), len(paths)))
            for path in paths_failed:
                logger.critical('{}'.format(path))
        else:
            logger.info('Finished with all files.')
        if settings['collate results csv to xlsx']:
            collate_results_csv_to_xlsx(path=result_folder, save_path=result_folder)  # folder_path

    if settings['shut down after analysis']:
        shutdown()
    logger.info('Elapsed time: {}\n{}\n'.format(elapsed_time(t_one), filler_for_logger))
    stop_logging_queue(logger, settings)
    listener.join()
    return paths_finished


if __name__ == '__main__':
    ysmr()
