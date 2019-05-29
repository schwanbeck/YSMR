#!/usr/bin/env python3
import logging
import multiprocessing as mp
import subprocess
import sys
from datetime import datetime
from time import sleep

import cv2  # opencv-contrib-python v3.4.5.20; needs numpy
import numpy as np  # needed by cv2; otherwise crashes/freezes without comment

from helper_file import (
    _backup,
    check_logfile,
    elapsed_time,
    find_paths,
    get_base_path,
    get_configs,
    get_loggers,
)
from like_a_record_baby import track_bacteria  # , start_it_up,

if __name__ == '__main__':
    t_one = datetime.now()  # to get rough time estimation
    # Set up stream and file handlers
    all_settings_dicts = get_configs()  # Get settings
    if all_settings_dicts is not None:
        default_settings, video_settings, _, eval_settings, test_settings = all_settings_dicts
    else:
        sys.exit('Fatal error in retrieving tracking.ini')
    check_logfile(path=default_settings['log_file'])
    queue_listener, format_for_logging = get_loggers(
        log_level=default_settings['logging_level'],
        logfile_name=default_settings['log_file'],
        use_short=default_settings['short_sys_log'])
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
    filler_for_logger = ''
    for sub_string in explain_logger_setup.split('\t'):  # create filler with '#' and correct tab placement
        filler_for_logger += '#' * len(sub_string) + '\t'
    filler_for_logger = filler_for_logger[:-1]  # remove last tab
    logger_main.info('Explanation\n{0}\n{1}\n{0}'.format(filler_for_logger, explain_logger_setup))

    # Warnings
    if default_settings['shut_down_after_analysis']:
        logger_main.warning('Shutting down PC after files have been processed')
    if test_settings['debugging']:
        logger_main.warning('Test settings enabled')
    if not cv2.useOptimized():
        logger_main.warning('Running cv2 unoptimised')
    if not eval_settings['evaluate_files']:
        logger_main.warning('Evaluation of .csv disabled')
    if not default_settings['store_old_list']:
        logger_main.warning('Old .csv result lists will be overwritten')
    if default_settings['delete_csv_afterwards']:
        logger_main.warning('Generated .csv files will be deleted after analysis')
    if default_settings['select_files']:
        if not test_settings['debugging']:
            logger_main.info('Manually selecting files enabled')
        else:
            logger_main.warning('Manually selecting files disabled due to test setting')
    # Infos
    if default_settings['verbose']:
        logger_main.info('Verbose enabled, logging set to debug.')
    else:
        logger_main.info('Log level set to {}'.format(default_settings['logging_level_name']))
    if video_settings['show_videos']:
        logger_main.info('Displaying videos')
    if video_settings['save_video']:
        logger_main.info('Saving detection video files')
    if video_settings['use_luminosity']:
        logger_main.info('Use average luminosity for distance calculation enabled - '
                         'processing time per video may increase notably')
    if eval_settings['limit_track_length']:  # 0 is false; otherwise true
        logger_main.info('Maximal track length for evaluation set to {} s'.format(eval_settings['limit_track_length']))
    else:
        logger_main.info('Full track length will be used in evaluation')
    try:
        default_settings['maximal_video_age'] = int(default_settings['maximal_video_age'])
        logger_main.debug('maximal video file age (infinite or seconds) set to {}'.format(
            default_settings['maximal_video_age']))
    except ValueError as max_vid_age_value_error:
        if default_settings['maximal_video_age'].lower() == 'infinite':
            logger_main.debug('maximal video file age (infinite or seconds) set to infinite')
        else:
            logger_main.exception(max_vid_age_value_error)
        default_settings['maximal_video_age'] = np.inf
    finally:
        pass
    # Debug messages
    logger_main.debug('White bacteria on dark background set to {}'.format(
        video_settings['white_bacteria_on_dark_background']))
    logger_main.debug('List save length set to {} entries'.format(default_settings['list_save_length_interval']))
    logger_main.debug('Pixel/micrometre: {}'.format(video_settings['pixel_per_micrometre']))

    if test_settings['debugging']:  # multiprocess can be uncommunicative with errors
        track_bacteria(test_settings['test_video'], settings_dicts=all_settings_dicts)

    else:
        pool = mp.Pool()  # get a pool of worker processes per available core
        if default_settings['select_files']:
            folder_path = get_base_path(rename=True)
            if folder_path is None:
                logger_main.warning('No valid path selected or error during path selection.\n')
                queue_listener.stop()
                sys.exit('No valid path selected or error during path selection.')
            paths = []
            if video_settings['use_default_extensions']:
                extensions = ['.avi', '.mp4', '.mov']
            else:
                extensions = []
            if video_settings['video_extension'] not in extensions:
                extensions.append(video_settings['video_extension'])
            ext_message = 'Looking for extensions ending in'
            for ext in extensions:
                ext_message += ' {},'.format(ext)
            logger_main.info(ext_message[:-1])  # get rid of trailing comma
            if not extensions:
                exit_warning = 'No extensions provided / found, please check settings \'video extension\' ' \
                               'and \'use default extensions (.avi, .mp4, .mov)\' in tracking.ini.\n'
                logger_main.critical(exit_warning)
                queue_listener.stop()
                sys.exit(exit_warning)

            for ext in extensions:
                paths.extend(find_paths(base_path=folder_path,
                                        extension=ext,
                                        minimal_age=default_settings['minimal_video_age'],
                                        maximal_age=default_settings['maximal_video_age'], ))
            # Remove generated output files
            paths = [path for path in paths if '_output.' not in path]
            if len(paths) <= 0:  # Might as well stop
                logger_main.warning('No acceptable files found in {}\n'.format(folder_path))
                queue_listener.stop()
                sys.exit('No files found in {}'.format(folder_path))
            paths.sort()
        else:
            paths = [test_settings['test_video']]
            logger_main.info('Test video path selected')
            # @todo: get video file list per calling args/argparser
        for path in paths:
            logger_main.debug(path)
        logger_main.info('Total number of files: {}'.format(len(paths)))
        # print('\nTotal number of files: {}'.format(len(paths)))

        while default_settings['user_input']:  # give user chance to check input
            logger_main.debug('Waiting for user input.')
            sleep(0.1)  # So the logger doesn't interfere with user input
            event = input('Continue? (Y/N): ')
            if 0 < len(event) < 4:
                if event[0].lower() == 'n':
                    logger_main.info('Process aborted.\n')
                    queue_listener.stop()
                    sys.exit('Process aborted.')
                elif event[0].lower() == 'y':
                    logger_main.debug('User has given it\'s blessing.')
                    break
        results = {}
        for path in paths:
            # Asynchronous calls to track_bacteria() with each path
            results[path] = pool.apply_async(track_bacteria, args=(path, all_settings_dicts,))
        pool.close()
        pool.join()

        paths_failed = []
        for path, item in results.items():
            try:
                result = item.get()
                if result is None:
                    paths_failed.append(path)
            except Exception as exc:
                logger_main.critical('An exception of type {0} occurred with path {1}. Arguments:'.format(
                    type(exc).__name__, path))
                for line in str(exc.args).splitlines():
                    logger_main.critical('{}'.format(line))
                logger_main.exception(exc)
                paths_failed.append(path)
                continue
            finally:
                pass
        if paths_failed:
            logger_main.critical('Failed to analyse {} of {} file(s):'.format(len(paths_failed), len(paths)))
            for path in paths_failed:
                logger_main.critical('{}'.format(path))
        else:
            logger_main.info('Finished with all files.')
    # @todo: remove _backup later
    _backup()

    if default_settings['shut_down_after_analysis']:
        if sys.platform == 'win32':
            try:
                subprocess.run('shutdown -f -s -t {}'.format(60))  # windows
                logger_main.warning('Calling \'shutdown -f -s -t {0}\' on system, shutting down in {0} s'.format(60))
                logger_main.info('Type \'shutdown -a\' in command console to abort shutdown.')
            except (OSError, FileNotFoundError) as os_shutdown_error:
                logger_main.exception('Error during shutdown: {}'.format(os_shutdown_error))
            finally:
                pass
        else:  # @todo: untested
            try:
                subprocess.run('systemctl poweroff')
                logger_main.warning('Calling \'systemctl poweroff\' on system.')
            except (OSError, FileNotFoundError) as os_shutdown_error:
                try:
                    subprocess.run('sudo shutdown -h +1')
                    logger_main.warning('Calling \'sudo shutdown -h +1\' on system.')
                except (OSError, FileNotFoundError) as os_shutdown_error:
                    logger_main.exception('Error during shutdown: {}'.format(os_shutdown_error))
            finally:
                pass
    logger_main.info('Elapsed time: {}\n{}\n\n'.format(elapsed_time(t_one), filler_for_logger))
    queue_listener.stop()
    sys.exit(0)
