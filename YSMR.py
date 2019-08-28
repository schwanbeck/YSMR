#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Copyright 2019 Julian Schwanbeck (julian.schwanbeck@med.uni-goettingen.de)
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
import sys
from datetime import datetime
from time import sleep

from helper_file import (check_logfile, collate_results_csv_to_xlsx, create_results_folder, elapsed_time, get_any_paths,
                         get_configs, get_loggers, log_infos, shutdown)
from track_eval import evaluate_tracks, select_tracks, track_bacteria


def analyse(path, settings=None, result_folder=None):
    """
    Starts analysis function (evaluate_tracks, select_tracks, track_bacteria), depending on received file type.
    Continues with analysis depending on provided settings.

    :param path: path to file
    :param settings: tracking.ini settings
    :type settings: dict
    :param result_folder: folder to save results in
    :return: pandas data frame(s), depending on last analysis step
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
        log_to_file=settings['log to file']
    )
    logger = logging.getLogger('ysmr').getChild(__name__)
    return_value = None
    if result_folder is None:
        result_folder = create_results_folder(path)
    # See if we need to evaluate for anything, otherwise we'll skip evaluate_tracks()
    plots_eval = any([
        settings['store generated statistical .csv file'],
        settings['save large plots'],
        settings['save rose plot'],
        settings['save time violin plot'],
        settings['save acr violin plot'],
        settings['save length violin plot'],
        settings['save turning point violin plot'],
        settings['save speed violin plot'],
        settings['save angle distribution plot / bins'],
        settings['collate results csv to xlsx'],
    ])
    df, fps, f_height, f_width, csv_file = [None] * 5

    while True:  # so we can break on error
        if '_statistics.csv' in path:  # Already evaluated file
            logger.warning('File already evaluated. '
                           'Please supply file ending in \'selected_data.csv\' instead. '
                           'File: {}'.format(path))
            return_value = None
            break
        if '.csv' not in path:  # as long as it's not a .csv, it should be a video:
            track_result = track_bacteria(video_path=path, settings=settings, result_folder=result_folder)
            if track_result is None:
                logger.warning('Error during video analysis of file {}.'.format(path))
                return_value = None
                break
            (df, fps, f_height, f_width, csv_file) = track_result
            return_value = df
        # if it's not evaluated yet:
        if 'selected_data.csv' not in path and (plots_eval or settings['store processed .csv file']):
            df = select_tracks(
                path_to_file=path,
                df=df,
                results_directory=result_folder,
                fps=fps,
                frame_height=f_height,
                frame_width=f_width,
                settings=settings
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
                fps=fps
            )
        # if nothing is selected for evaluation and it's specifically a selected_data.csv, something seems wrong
        elif 'selected_data.csv' in path:
            logger.warning('No evaluation set to true in settings. Did not evaluate {}'.format(path))
        break  # all is well
    if settings['delete .csv file after analysis'] and csv_file:
        try:
            os.remove(csv_file)
        except FileNotFoundError:
            pass
        except Exception as ex:
            template = 'An exception of type {0} occurred. Arguments:\n{1!r}'
            logger.exception(template.format(type(ex).__name__, ex.args))
    if return_value:
        end_string = 'Finished with'
    else:
        end_string = 'Error during'
    logger.info('{} process - module: {} PID: {}, elapsed time: {}'.format(
        end_string, __name__, os.getpid(), elapsed_time(t_one)))
    return return_value


def ysmr(settings=None, paths=None):
    """
    Starts asynchronous multiprocessing of provided file(s) with analyse().

    :param settings: tracking.ini settings
    :type settings: dict
    :param paths: path or iterable with paths
    :rtype paths: str, list, os.PathLike
    :return: list of finished paths, list of failed paths
    :rtype paths_finished: list
    :rtype paths_failed: list
    """
    t_one = datetime.now()  # to get rough time estimation
    settings = get_configs(settings)  # Get settings
    paths_failed = []
    paths_finished = []
    if isinstance(paths, str) or isinstance(paths, os.PathLike):
        paths = [paths]  # convert to list, otherwise for path in paths iterates over characters in string
    if settings is None:
        sys.exit('Fatal error in retrieving tracking.ini')
    check_logfile(path=settings['log file path'])
    queue_listener, format_for_logging = get_loggers(
        log_level=settings['log_level'],
        logfile_name=settings['log file path'],
        short_stream_output=settings['shorten displayed logging output'],
        short_file_output=settings['shorten logfile logging output'],
        log_to_file=settings['log to file']
    )
    logger = logging.getLogger('ysmr').getChild(__name__)
    filler_for_logger = log_infos(settings=settings, format_for_logging=format_for_logging)

    if settings['debugging']:  # multiprocess can be uncommunicative with errors
        folder_path = os.path.dirname(settings['path to test video'])
        result_folder = create_results_folder(path=folder_path)
        result = analyse(settings['path to test video'], settings=settings, result_folder=result_folder)
        if result:
            paths_finished.append(settings['path to test video'])
        else:
            paths_failed.append(settings['path to test video'])

    else:
        if settings['select files']:
            if not paths:
                paths = get_any_paths(rename=True)
            if not paths:
                logger.critical('No files selected.')
                queue_listener.stop()
                sys.exit('No files selected.')
        else:
            if not paths:
                paths = [settings['path to test video']]
            logger.info('Test video path selected')
        folder_path = os.path.dirname(paths[0])
        for path in paths:
            logger.debug(path)
        logger.info('Total number of files: {}'.format(len(paths)))

        while settings['user input']:  # give user chance to check input
            logger.debug('Waiting for user input.')
            sleep(0.1)  # So the logger doesn't interfere with user input
            event = input('Continue? (Y/N): ')
            if 0 < len(event) < 4:
                if event[0].lower() == 'n':
                    logger.info('Process aborted.\n')
                    queue_listener.stop()
                    sys.exit('Process aborted.')
                elif event[0].lower() == 'y':
                    logger.debug('User has given it\'s blessing.')
                    break
        results = {}
        result_folder = create_results_folder(paths[0])

        pool = mp.Pool()  # get a pool of worker processes per available core
        for path in paths:
            # Asynchronous calls to track_bacteria() with each path
            results[path] = pool.apply_async(analyse, args=(path, settings, result_folder))
        pool.close()
        pool.join()

        for path, item in results.items():
            try:
                result = item.get()
                if result is None:
                    paths_failed.append(path)
                else:
                    paths_finished.append(path)
            except Exception as exc:
                logger.critical('An exception of type {0} occurred with path {1}. Arguments:'.format(
                    type(exc).__name__, path))
                for line in str(exc.args).splitlines():
                    logger.critical('{}'.format(line))
                logger.exception(exc)
                paths_failed.append(path)
                continue
        if paths_failed:
            logger.critical('Failed to analyse {} of {} file(s):'.format(len(paths_failed), len(paths)))
            for path in paths_failed:
                logger.critical('{}'.format(path))
        else:
            logger.info('Finished with all files.')
        if settings['collate results csv to xlsx']:
            collate_results_csv_to_xlsx(path=folder_path, save_path=result_folder)

    if settings['shut down after analysis']:
        shutdown()
    logger.info('Elapsed time: {}\n{}\n'.format(elapsed_time(t_one), filler_for_logger))
    queue_listener.stop()
    return paths_finished, paths_failed


if __name__ == '__main__':
    ysmr()
