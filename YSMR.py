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

import cv2  # needs numpy
import numpy as np  # needed by cv2; otherwise crashes/freezes without comment

from helper_file import (
    _backup,
    check_logfile,
    collate_results_csv_to_xlsx,
    create_results_folder,
    elapsed_time,
    find_paths,
    get_base_path,
    get_configs,
    get_loggers,
    shutdown)
from like_a_record_baby import track_bacteria  # , start_it_up,

if __name__ == '__main__':
    t_one = datetime.now()  # to get rough time estimation
    # Set up stream and file handlers
    settings = get_configs()  # Get settings
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
    # Log some general stuff
    logger = logging.getLogger('ei').getChild(__name__)
    explain_logger_setup = format_for_logging.format(**{
        'asctime': 'YYYY-MM-DD HH:MM:SS,mmm',  # ISO8601 'YYYY-MM-DD HH:MM:SS+/-TZ'
        'name': 'logger name',
        'funcName': 'function name',
        'filename': 'file name',
        'lineno': 'lNr',
        'levelname': 'level',
        'process': 'PID',
        'message': 'Message (lNr: line number, PID: Process ID)'
    })
    filler_for_logger = ''
    for sub_string in explain_logger_setup.split('\t'):  # create filler with '#' and correct tab placement
        filler_for_logger += '#' * len(sub_string) + '\t'
    filler_for_logger = filler_for_logger[:-1]  # remove last tab
    logger.info('Explanation\n{0}\n{1}\n{0}'.format(filler_for_logger, explain_logger_setup))

    # Warnings
    if settings['shut down after analysis']:
        logger.warning('Shutting down PC after files have been processed')
    if settings['debugging']:
        logger.warning('Test settings enabled')
    if not cv2.useOptimized():
        logger.warning('Running cv2 unoptimised')
    if not settings['rename previous result .csv']:
        logger.warning('Old .csv result lists will be overwritten')
    if settings['delete .csv file after analysis']:
        logger.warning('Generated .csv files will be deleted after analysis')
    if settings['select files']:
        if not settings['debugging']:
            logger.info('Manually selecting files enabled')
        else:
            logger.warning('Manually selecting files disabled due to test setting')
    # Infos
    if settings['verbose']:
        logger.info('Verbose enabled, logging set to debug.')
    else:
        logger.info('Log level set to {}'.format(settings['set logging level (debug/info/warning/critical)']))
    if settings['display video analysis']:
        logger.info('Displaying videos')
    if settings['save video']:
        logger.info('Saving detection video files')
    if settings['include luminosity in tracking calculation']:
        logger.info('Use average luminosity for distance calculation enabled - '
                    'processing time per video may increase notably')
    if settings['limit track length to x seconds']:  # 0 is false; otherwise true
        logger.info('Maximal track length for evaluation set to {} s'.format(
            settings['limit track length to x seconds']))
    else:
        logger.info('Full track length will be used in evaluation')
    if settings['maximal video file age (infinite or seconds)'] == np.inf:
        logger.debug('maximal video file age (infinite or seconds) set to infinite')
    else:
        logger.debug('maximal video file age (infinite or seconds) set to {}'.format(
            settings['maximal video file age (infinite or seconds)']))

    # Debug messages
    logger.debug('White bacteria on dark background set to {}'.format(
        settings['white bacteria on dark background']))
    logger.debug('List save length set to {} entries'.format(settings['list save length interval']))
    logger.debug('Pixel/micrometre: {}'.format(settings['pixel per micrometre']))

    # if settings['result folder'] != 'None':

    if settings['debugging']:  # multiprocess can be uncommunicative with errors
        folder_path = os.path.dirname(settings['path to test video'])
        result_folder = create_results_folder(path=folder_path)
        track_bacteria(settings['path to test video'], settings=settings, result_folder=result_folder)

    else:
        pool = mp.Pool()  # get a pool of worker processes per available core
        if settings['select files']:
            folder_path = get_base_path(rename=True)
            if folder_path is None:
                logger.warning('No valid path selected or error during path selection.\n')
                queue_listener.stop()
                sys.exit('No valid path selected or error during path selection.')
            paths = []
            if settings['use default extensions (.avi, .mp4, .mov)']:
                extensions = ['.avi', '.mp4', '.mov']
            else:
                extensions = []
            if settings['video extension'] not in extensions:
                extensions.append(settings['video extension'])
            ext_message = 'Looking for extensions ending in'
            for ext in extensions:
                ext_message += ' {},'.format(ext)
            logger.info(ext_message[:-1])  # get rid of trailing comma
            if not extensions:
                exit_warning = 'No extensions provided / found, please check settings \'video extension\' ' \
                               'and \'use default extensions (.avi, .mp4, .mov)\' in tracking.ini.\n'
                logger.critical(exit_warning)
                queue_listener.stop()
                sys.exit(exit_warning)

            for ext in extensions:
                paths.extend(find_paths(base_path=folder_path,
                                        extension=ext,
                                        minimal_age=settings['minimal video file age in seconds'],
                                        maximal_age=settings['maximal video file age (infinite or seconds)'], ))
            # Remove generated output files
            paths = [path for path in paths if '_output.' not in path]
            if not paths:  # Might as well stop, [] == False
                logger.warning('No acceptable files found in {}\n'.format(folder_path))
                queue_listener.stop()
                sys.exit('No files found in {}'.format(folder_path))
            paths.sort()
        else:
            paths = [settings['path to test video']]
            folder_path = os.path.dirname(settings['path to test video'])
            logger.info('Test video path selected')
            # @todo: get video file list per calling args/argparser
        for path in paths:
            logger.debug(path)
        logger.info('Total number of files: {}'.format(len(paths)))
        # print('\nTotal number of files: {}'.format(len(paths)))

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
        for path in paths:
            # Asynchronous calls to track_bacteria() with each path
            results[path] = pool.apply_async(track_bacteria, args=(path, settings, result_folder))
        pool.close()
        pool.join()

        paths_failed = []
        for path, item in results.items():
            try:
                result = item.get()
                if result is None:
                    paths_failed.append(path)
            except Exception as exc:
                logger.critical('An exception of type {0} occurred with path {1}. Arguments:'.format(
                    type(exc).__name__, path))
                for line in str(exc.args).splitlines():
                    logger.critical('{}'.format(line))
                logger.exception(exc)
                paths_failed.append(path)
                continue
            finally:
                pass
        if paths_failed:
            logger.critical('Failed to analyse {} of {} file(s):'.format(len(paths_failed), len(paths)))
            for path in paths_failed:
                logger.critical('{}'.format(path))
        else:
            logger.info('Finished with all files.')
        if settings['collate results csv to xlsx']:
            collate_results_csv_to_xlsx(path=folder_path, save_path=result_folder)
    # @todo: remove _backup later
    _backup()

    if settings['shut down after analysis']:
        shutdown()
    logger.info('Elapsed time: {}\n{}\n'.format(elapsed_time(t_one), filler_for_logger))
    queue_listener.stop()
    sys.exit(0)
