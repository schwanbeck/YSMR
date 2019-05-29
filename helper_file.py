#!/usr/bin/env python3
import configparser
import logging
import os
import platform
import shutil
import sys
from datetime import datetime, timedelta
from glob import glob
from itertools import cycle as cycle
from logging.handlers import QueueHandler, QueueListener
from queue import Queue
from tkinter import filedialog, Tk

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import argrelextrema

_config = configparser.ConfigParser(allow_no_value=True)


def argrelextrema_groupby(group, comparator=np.less_equal, order=15, shift=False, shift_range=4):
    result = np.zeros(group.shape[0], dtype=np.int8)
    # @todo: changed .values to .array
    np.put(result, argrelextrema(group.array, comparator, order=order)[0], 1)
    if shift:
        result_comp = result
        for d_shift in range(-1, -(shift_range + 1)):
            query = shift_np_array(result_comp, d_shift, 0)
            result = np.where((
                (result == 1) &
                (query == 1),
                0, result))
    group = np.where(result == 1, group.array, 0)
    return group


def bytes_to_human_readable(number_of_bytes):
    """
    Inspired by https://stackoverflow.com/questions/44096473/
    Returns string containing bytes, rounded to 1 decimal place,
    with unit prefix as defined by SI.
    For documentation purposes only.
    Author: Julian Schwanbeck
    """
    if number_of_bytes < 0:  # assert number_of_bytes >= 0, 'Negative bytes passed'
        return 'Negative Bytes'  # As this isn't / shouldn't be used anywhere important, this suffices
    bytes_in_a_kb = 1024  # 2**10, as it should be
    units = ['bytes', 'KB', 'MB', 'GB', 'TB', 'PB', 'EB', 'ZB', 'YB']  # Update as needed
    for unit in units:  # Go through each unit
        if number_of_bytes / bytes_in_a_kb < 1 or unit == units[-1]:
            break  # stop if division would go below 1 or end of list is reached
        number_of_bytes /= bytes_in_a_kb  # divide otherwise
    return '{0:.01f} {1}'.format(number_of_bytes, unit)


def create_configs():
    logger = logging.getLogger('ei').getChild(__name__)
    logger.warning('tracking.ini was reset to default values')
    _config['DEFAULT_SETTINGS'] = {
        'user input': True,
        'shut down after analysis': False,
        'select files': True,
        'verbose': False,
        'rename previous result .csv': False,
        'delete .csv file after analysis': True,
        'previous directory': './',
        'list save length interval': 10000,
        # see below
        'time since creation in seconds': 0,
        'maximal video file age (infinite or seconds)': 'infinite',
        'set logging level (debug/info/warning/critical)': 'debug',
        'shorten displayed logging output': False,
        'log file path': './logfile.log',
    }
    _config.set('DEFAULT_SETTINGS',
                '# color filter must be conversion to gray value. Set set_color_filter to True in '
                'order to change. Also accepts integer value of flag.')
    _config.set('DEFAULT_SETTINGS',
                '# list save length: interval in which results are saved to disc, optimum is '
                'system/hardware dependent. Fiddle around if memory or performance is an issue. '
                'Saves at maximum once per frame; if this is an issue move save block in '
                'track_bacteria() into object.items() loop')
    _config.set('DEFAULT_SETTINGS',
                '# set logging level options: debug, info, warning, critical')

    # _config['DETECTION_SETTINGS'] = {}

    _config['VIDEO_SETTINGS'] = {
        'frames per second': 30,
        'pixel per micrometre': 1.41888781,
        'frame height': 922,
        'frame width': 1228,
        'minimal frame count': 600,
        'threshold offset for detection': 5,
        'video extension': '.mp4',
        'use default extensions (.avi, .mp4, .mov)': True,
        'white bacteria on dark background': True,
        'include luminosity in tracking calculation': False,
        'show videos': False,
        'save video': False,
        'set color filter': False,
        'color filter': 'COLOR_BGR2GRAY',
    }
    _config['TRACK_SELECTION'] = {
        'size outliers lower end in px': 2,
        'size outliers upper end in px': 50,
        'exclude measurement when above x times average area': 1.5,
        'minimal length in seconds': 20,
        'maximal consecutive holes': 5,
        'maximal empty frames in %': 5,
        'percent quantiles excluded area': 10,
        'try to omit motility outliers': True,
        'stop excluding motility outliers if total count above percent (0 for off)': 5,
        'average width/height ratio min.': 0.125,
        'average width/height ratio max.': 0.67,
        'percent of screen edges to exclude': 5,
        'maximal recursion depth (0 is off)': 960,
    }
    _config['EVALUATION_SETTINGS'] = {
        'evaluate files after analysis': True,
        'show large plots': False,
        'limit track length to seconds (0 is off)': 20,
        'limit track length exactly': False,
        'compare angle between n frames': 10,
        'min. angle in degrees for turning point': 30,
    }
    _config.set('EVALUATION_SETTINGS',
                '# Limit track length: 0 for off; will otherwise take set frames for analysis, '
                'or maximum track length, whichever is shorter.')
    _config['TEST_SETTINGS'] = {
        'debugging': False,
        'path to test video': 'Q:/test_video.avi',
        'path to test .csv': 'Q:/test_list.csv',
        'last backup': '2019-03-29 13:37:30.330330',
    }
    try:
        with open('tracking.ini', 'w+') as configfile:
            _config.write(configfile)
    except (IOError, OSError) as configfile_error:
        logger.exception('Could not create config file: {}'.format(configfile_error))


# Check tracking.ini
if not os.path.isfile('tracking.ini'):  # needed by later functions
    create_configs()
try:  # Check if all sections are present and accessible
    _config.read('tracking.ini')
    for _c_key in ['DEFAULT_SETTINGS', 'VIDEO_SETTINGS', 'TRACK_SELECTION', 'EVALUATION_SETTINGS', 'TEST_SETTINGS', ]:
        _ = _config[_c_key]
except KeyError:  # create configs anew otherwise
    try:
        os.rename('tracking.ini', '{}_tracking.ini.old'.format(datetime.now().strftime('%y%m%d%H%M%S')))
    finally:
        pass
    create_configs()
    _config.read('tracking.ini')
_default = _config['DEFAULT_SETTINGS']


def _backup(skip_check_time=False, time_delta_days=0, time_delta_hours=20, time_delta_min=0):
    logger = logging.getLogger('ei').getChild(__name__)
    last_backup = _config['TEST_SETTINGS'].get('last backup')
    last_backup = datetime.strptime(last_backup, '%Y-%m-%d %H:%M:%S.%f')
    now_utc = datetime.utcnow()
    now = datetime.now()
    diff = timedelta(days=time_delta_days, hours=time_delta_hours, minutes=time_delta_min)
    if (now_utc - last_backup) > diff or skip_check_time:
        logger.info('Creating backup of program')
        src = os.getcwd()
        names = os.listdir(src)
        now_folder = now.strftime('%y%m%d%H%M%S')  # y%m%d%H%M%S
        dst = 'Q:/Code/{}'.format(now_folder)
        _mkdir(dst)
        ignore = shutil.ignore_patterns(
            '~$*', '._sync*', '.owncloud*', 'Thumbs.db', '*.tmp', 'desktop.ini',
            '*.partial', '_conflict-*', 'FolderStatistic', 'ignore.patterns', 'Last_Sync',
            '.PowerFolder*', '(downloadmeta)*',
        )
        if ignore is not None:
            ignored_names = ignore(src, names)
        else:
            ignored_names = set()

        for name in names:
            src_name = os.path.join(src, name)
            dst_name = os.path.join(dst, name)
            if name in ignored_names:
                continue
            if os.path.isdir(name):
                continue
            try:
                shutil.copy2(src_name, dst_name)
            except (IOError, os.error, shutil.Error) as why:
                error = '{2}: {0}, {1}'.format(src_name, dst_name, str(why))
                logger.debug(error)
        if not skip_check_time:
            _config.set('TEST_SETTINGS', 'last backup', str(now_utc))
            with open('tracking.ini', 'w') as configfile:
                _config.write(configfile)
            logger.debug('Previous backup set to {} (lokal time: {})'.format(now_utc, now))


def check_logfile(path, max_size=2 ** 20):  # max_size=1 MB
    # RotatingFileHandler does the same but can't do the
    # rollover when used with multiprocess, so we create our own approximation
    if os.path.isfile(path):
        file_size = os.path.getsize(path)
    else:
        file_size = 0
    if file_size < max_size:
        return path
    base_path, file_name = os.path.split(path)
    old_paths = find_paths(base_path=base_path, extension='{}.*'.format(file_name), recursive=False)
    if old_paths:
        old_paths = sorted(old_paths, reverse=True, key=lambda x: int(x[-1]))
        counts = [int(count[-1]) for count in old_paths]
        if not counts[-1] > 1:
            max_idx = [1]
            max_idx.extend([s - t for s, t in zip(counts[:-1], counts[1:])])
            max_idx = np.array(max_idx).argmax()
            for old_count, old_path in zip(counts[max_idx:], old_paths[max_idx:]):
                try:
                    if old_count == 9:
                        os.remove(old_path)
                    else:
                        new_path = '{}{}'.format(old_path[:-1], old_count + 1)
                        if not os.path.isfile(new_path):
                            os.rename(old_path, new_path)
                finally:
                    pass
    try:
        os.rename(path, '{}.1'.format(path))
    finally:
        return path


def creation_date(path_to_file):
    """
    Try to get the date that a file was created, falling back to when it was
    last modified if that isn't possible.
    See for explanation/origin:
    https://stackoverflow.com/a/39501288/1709587
    Accessed last 2019-03-01 13:37:00,101
    """
    if os.path.isfile(path_to_file):
        now = datetime.now()
        if platform.system() == 'Windows':
            then = os.path.getctime(path_to_file)
        else:
            stat = os.stat(path_to_file)
            try:
                then = stat.st_birthtime
            except AttributeError:
                # We're probably on Linux. No easy way to get creation dates here,
                # so we'll settle for when its content was last modified.
                then = stat.st_mtime
        time_delta = now - datetime.fromtimestamp(then)
        seconds = time_delta.total_seconds()
        return seconds
    else:
        return None


def different_tracks(data, column='TRACK_ID'):
    # search for individual tracks; return lists of starts/stops
    track_id = data[column].array
    # get np.ndarray with the values where track changes occur
    # by comparing every element with the previous one
    changes = np.where(track_id[:-1] != track_id[1:])[0]
    changes = changes.tolist()  # convert to list for manipulation
    starts = [0]  # Initialise starts - track 0 starts at index 0 obviously
    starts.extend([item + 1 for item in changes])  # others start at stop + 1
    changes.append(data.index.max())  # Append last stop to list of stops
    return starts, changes


def elapsed_time(time_one):
    logger = logging.getLogger('ei').getChild(__name__)
    time_two = datetime.now()
    try:
        time_delta = time_two - time_one
    except ValueError as val_error:
        logger.exception(val_error)
        return None
    return time_delta


def find_paths(base_path=None, extension=None, minimal_age=0, maximal_age=np.inf, recursive=True):
    logger = logging.getLogger('ei').getChild(__name__)
    if not os.path.exists(base_path):
        logger.critical('Path could not be found: {}'.format(base_path))
        return None
    if extension is None:
        extension = _config['VIDEO_SETTINGS'].get('extension', fallback='.mp4')
    # if extension[0] != '.' and len(extension) == 3:
    #     extension = '.{}'.format(extension)
    if minimal_age is None:
        minimal_age = _default.getint('time since creation in seconds', fallback=0)
    if base_path is None:
        base_path = 'Q:/Movies/{}{}'.format('**/*', extension)
    else:
        if base_path[-1] != '/':
            base_path = '{}/'.format(base_path)
        base_path = '{}**/*{}'.format(base_path, extension)
    in_files = glob(base_path, recursive=recursive)
    out_files = []
    for file in in_files:
        file = file.replace(os.sep, '/')
        # checks if file is young/old enough
        file_creation_date = creation_date(file)
        if file_creation_date >= 0 or minimal_age < 0:
            if maximal_age >= file_creation_date >= minimal_age:
                # Set path separators to /
                out_files.append(file)
        else:
            logger.warning('The file appears to be {:.2f} seconds '
                           'from the future and was thus not selected. '
                           'To circumvent this, set minimal age in tracking.ini'
                           ' to a negative value. '
                           'File: {}'.format(abs(file_creation_date), file))
    del in_files
    return out_files


def get_base_path(rename=False, prev_dir=None):
    # @todo: get PyQT instead?
    logger = logging.getLogger('ei').getChild(__name__)
    if prev_dir is None:
        prev_dir = _default.get('previous directory', fallback='./')
    try:
        root = Tk()
        root.overrideredirect(1)  # hide the root window
        root.withdraw()
        curr_path = filedialog.askdirectory(
            title='Choose a base directory to look for files. ',
            mustexist=True,
            initialdir=prev_dir
        )
    except Exception as ex:
        template = 'An exception of type {0} occurred. Arguments:\n{1!r}'
        logger.exception(template.format(type(ex).__name__, ex.args))
        return None
    if not os.path.isdir(curr_path):
        logger.warning('No Path selected. ')
        return None
    if rename:
        _config.set('DEFAULT_SETTINGS', 'previous directory', curr_path)
        with open('tracking.ini', 'w') as configfile:
            _config.write(configfile)
        logger.debug('Previous directory set to {}'.format(curr_path))
    else:
        logger.debug('Selected path: {}'.format(curr_path))
    return curr_path


def get_colour_map(counts, colour_map=plt.cm.gist_rainbow):
    return cycle(reversed([colour_map(col / counts) for col in range(0, counts + 1, 1)]))


def get_configs(first_try=True):
    logger = logging.getLogger('ei').getChild(__name__)
    settings_dicts = None
    try:
        default = _config['DEFAULT_SETTINGS']
        verbose = default.getboolean('verbose')
        set_log_level = default.get('set logging level (debug/info/warning/critical)')
        log_levels = {'debug': logging.DEBUG,
                      'info': logging.INFO,
                      'warning': logging.WARNING,
                      'critical': logging.CRITICAL}
        set_log_level_setting = logging.DEBUG  # fallback
        if not verbose:  # if verbose logging is set to DEBUG
            if set_log_level.lower() in log_levels.keys():
                set_log_level_setting = log_levels[set_log_level.lower()]
            else:
                logger.warning('Logging level passed argument: {}. '.format(set_log_level) +
                               'Argument not recognised. Logging set to debug. Accepted arguments: ' +
                               ''.join('{} '.format(format_level) for format_level in log_levels.keys()))
        default_settings_dict = {
            'user_input': default.getboolean('user input'),
            'shut_down_after_analysis': default.getboolean('shut down after analysis'),
            'select_files': default.getboolean('select files'),
            'verbose': verbose,
            'store_old_list': default.getboolean('rename previous result .csv'),
            'delete_csv_afterwards': default.getboolean('delete .csv file after analysis'),
            'previous_directory': default.get('previous directory', fallback='./'),
            'list_save_length_interval': default.getint('list save length interval'),
            'minimal_video_age': default.getint('time since creation in seconds'),
            'maximal_video_age': default.get('maximal video file age (infinite or seconds)'),
            'logging_level': set_log_level_setting,
            'logging_level_name': set_log_level,
            'short_sys_log': default.getboolean('shorten displayed logging output'),
            'log_file': default.get('log file path')
        }

        video_settings = _config['VIDEO_SETTINGS']
        set_different_colour_filter_bool = video_settings.getboolean('set_color_filter', fallback=False)
        colour_filter = cv2.COLOR_BGR2GRAY
        if set_different_colour_filter_bool:
            colour_filter = set_different_colour_filter(video_settings.get('color filter', fallback='COLOR_BGR2GRAY'))

        video_settings_dict = {
            'frames_per_second': video_settings.getfloat('frames per second'),
            'pixel_per_micrometre': video_settings.getfloat('pixel per micrometre'),
            'frame_height': video_settings.getint('frame height'),
            'frame_width': video_settings.getint('frame width'),
            'min_frame_count': video_settings.getint('minimal frame count'),
            'threshold_offset': video_settings.getint('threshold offset for detection'),
            'video_extension': video_settings.get('video extension'),
            'use_default_extensions':
                video_settings.getboolean('use default extensions (.avi, .mp4, .mov)', fallback=True),
            'white_bacteria_on_dark_background':
                video_settings.getboolean('white bacteria on dark background'),
            'use_luminosity': video_settings.getboolean('include luminosity in tracking calculation'),
            'show_videos': video_settings.getboolean('show videos'),
            'save_video': video_settings.getboolean('save video'),
            'set_color_filter': set_different_colour_filter_bool,
            'color_filter': colour_filter,
        }

        track_settings = _config['TRACK_SELECTION']
        track_selection_dict = {
            'max_average_area': track_settings.getfloat('exclude measurement when above x times average area'),
            'min_px': track_settings.getfloat('size outliers lower end in px'),
            'max_px': track_settings.getfloat('size outliers upper end in px'),
            'min_size': track_settings.getint('minimal length in seconds'),
            'max_holes': track_settings.getint('maximal consecutive holes'),
            'max_duration_size_ratio': (track_settings.getfloat('maximal empty frames in %', fallback=5) / 100 + 1),
            'omit_motility_outliers': track_settings.getboolean('try to omit motility outliers'),
            'motility_outliers_max':
                track_settings.getfloat('stop excluding motility outliers if total count above percent (0 for off)')
                / 100,
            'cutoff_area_quantile': (track_settings.getfloat('percent quantiles excluded area', fallback=10) / 100),
            'min_size_ratio': track_settings.getfloat('average width/height ratio min.'),
            'max_size_ratio': track_settings.getfloat('average width/height ratio max.'),
            'frame_exclusion_percentage':
                (track_settings.getfloat('percent of screen edges to exclude', fallback=5) / 100),
            'max_recursion_depth': track_settings.getint('maximal recursion depth (0 is off)'),
        }

        eval_settings = _config['EVALUATION_SETTINGS']
        eval_settings_dict = {
            'evaluate_files': eval_settings.getboolean('evaluate files after analysis'),
            'limit_track_length': eval_settings.getint('limit track length to seconds (0 is off)'),
            'limit_track_length_exact': eval_settings.getboolean('limit track length exactly'),
            'large_plots': eval_settings.getboolean('show large plots'),
            'min_angle': eval_settings.getfloat('min. angle in degrees for turning point'),
            'angle_diff': eval_settings.getint('compare angle between n frames'),
        }

        test_settings = _config['TEST_SETTINGS']
        test_settings_dict = {
            'debugging': test_settings.getboolean('debugging'),
            'test_video': test_settings.get('path to test video'),
            'test_csv': test_settings.get('path to test .csv')
        }
        settings_dicts = [default_settings_dict, video_settings_dict,
                          track_selection_dict, eval_settings_dict, test_settings_dict]
        for settings_dict in settings_dicts:  # check for missing values
            if verbose:
                logger.debug('tracking.ini settings:')
            for test_key in settings_dict:
                if settings_dict[test_key] is None:
                    error = 'tracking.ini is missing a value in {}'.format(test_key)
                    logger.critical(error)
                    settings_dicts = None
                    break
                elif verbose:
                    logger.debug('{}: {}'.format(test_key, settings_dict[test_key]))
    except Exception as ex:
        template = 'An exception of type {0} occurred while attempting to read tracking.ini. Arguments:'
        logger.critical(template.format(type(ex).__name__, ))
        for line in str(ex.args).splitlines():
            logger.critical('{}'.format(line))
    finally:
        pass
    if first_try and not settings_dicts:
        try:
            old_tracking_ini = '{}_tracking.ini.old'.format(datetime.now().strftime('%y%m%d%H%M%S'))
            os.rename('tracking.ini', old_tracking_ini)
            logger.critical('Old tracking.ini renamed to {}'.format(old_tracking_ini))
        finally:
            create_configs()
            return get_configs(first_try=False)  # avoid infinite loop
    elif not first_try and not settings_dicts:
        logger.critical('Fatal: Could not access or restore tracking.ini')
    return settings_dicts


def get_loggers(log_level=logging.DEBUG, logfile_name='./logfile.log', use_short=False):
    # The loggers name is "ei". This is german for "egg". There is no good reason for this.
    logger = logging.getLogger('ei')
    logger.propagate = False
    # Log message setup
    format_for_logging = '{asctime:}\t' \
                         '{name:21.21}\t' \
                         '{funcName:14.14}\t' \
                         '{lineno:>4}\t' \
                         '{levelname:8.8}\t' \
                         '{process:>5}:\t' \
                         '{message}'
    short_format = logging.Formatter('{asctime:},{msecs:03.0f} {levelname:} {process:}:\n{message}', '%H:%M:%S',
                                     style='{')
    # Sets the global logging format.
    logging.basicConfig(format=(format_for_logging, "%Y-%m-%dT%H:%M:%S%z"), style='{')
    queue_listener = None
    if len(logger.handlers) > 0:
        for handler in logger.handlers:
            if isinstance(handler, QueueHandler):
                queue_listener = handler  # if we have our handler, we can stop
                break
    if not queue_listener:  # otherwise, we have to set it up
        logger_formatter = logging.Formatter(format_for_logging, style='{')  # ISO8601: , "%Y-%m-%dT%H:%M:%S%z"
        stream_handler_logger = logging.StreamHandler(sys.stdout)
        if not use_short:
            stream_handler_logger.setFormatter(logger_formatter)
        else:
            stream_handler_logger.setFormatter(short_format)
        file_handler_logger = logging.FileHandler(filename=logfile_name, mode='a', )
        file_handler_logger.setFormatter(logger_formatter)

        # Set up log queue, add all listeners to queue
        log_queue = Queue(-1)
        queue_handler = QueueHandler(log_queue)
        logger.addHandler(queue_handler)
        queue_listener = QueueListener(log_queue, stream_handler_logger, file_handler_logger)
        queue_listener.start()
        logger.setLevel(log_level)
        stream_handler_logger.setLevel(log_level)
        file_handler_logger.setLevel(log_level)
    return queue_listener, format_for_logging


def get_data(csv_file_path, dtype=None):
    logger = logging.getLogger('ei').getChild(__name__)
    if type(csv_file_path) is not (str or os.PathLike or bytes) and (list or tuple):
        csv_file_path = csv_file_path[0]
        # hasattr(csv_file_path, '__')
        logger.warning('Passed list or tuple argument to get_data(); get_data() only used first argument.')
    try:
        file_size = os.path.getsize(csv_file_path)
        file_size = bytes_to_human_readable(file_size)
        logger.info('Reading file with size {}: {} '.format(file_size, csv_file_path))
    except (ValueError, TypeError) as size_error:
        logger.exception('Have accepted file {}; error during size reading: {}'.format(csv_file_path, size_error))
    finally:
        pass
    if dtype is None:
        dtype = {'TRACK_ID': np.uint32,
                 'POSITION_T': np.uint32,
                 # up to ~4 * 10**9 data points; value must be pos.
                 'POSITION_X': np.float64,
                 'POSITION_Y': np.float64,
                 'WIDTH': np.float64,
                 'HEIGHT': np.float64,
                 # 'DEGREES_ANGLE': np.float64
                 }
    use_cols = list(dtype.keys())
    try:
        # Fixes some special character problems with pd.read_csv paths:
        with open(csv_file_path, 'r', newline='\n') as csv:
            # csv_chunks =  # use chunks in case file is too large
            # Done automatically by pd.read_csv()
            data = pd.read_csv(csv,
                               sep=',',  # as is default
                               header=0,  # as is default
                               usecols=use_cols,
                               dtype=dtype,
                               )
    except ValueError as val_error:
        logger.exception(val_error, '\n Error: Invalid file type: {}'.format(csv_file_path))
        return None
    except OSError as makedir_error:
        logger.exception(makedir_error)
        return None
    logger.debug('Done reading {} into data frame'.format(csv_file_path))
    return data


def _mkdir(new_directory):
    """works the way a good mkdir should :)
        - already exists, silently complete
        - regular file in the way, raise an exception
        - parent directory(ies) does not exist, make them as well
        - https://code.activestate.com/recipes/82465-a-friendly-mkdir/
    """
    if os.path.isdir(new_directory):
        pass
    elif os.path.isfile(new_directory):
        raise OSError("a file with the same name as the desired dir, '%s', already exists." % new_directory)
    else:
        head, tail = os.path.split(new_directory)
        if head and not os.path.isdir(head):
            _mkdir(head)
        if tail:
            os.mkdir(new_directory)


def reshape_result(tuple_of_tuples, *args):
    (x, y), (w, h), degrees_orientation = tuple_of_tuples  # ((x, y), (w, h), additional_info), xy is centroid
    additional_info = (w, h, degrees_orientation)
    coordinates = [x, y]
    coordinates.extend(args)
    return tuple(coordinates), additional_info


def save_list(file_path, filename, coords=None, get_name=False, first_call=False, store_old_list=True):
    logger_save_list = logging.getLogger('ei.' + __name__)
    file_csv = '{}/{}_list.csv'.format(file_path, filename)
    if get_name:
        return file_csv  # return name, stop function

    if first_call:  # set up .csv file
        old_list = False
        if os.path.isfile(file_csv):
            if store_old_list:
                now = datetime.now().strftime('%y%m%d%H%M%S')
                old_filename, old_file_extension = os.path.splitext(file_csv)
                old_list = '{}_{}{}'.format(old_filename, now, old_file_extension)
                os.rename(file_csv, old_list)  # rename file
                logger_save_list.info('Renaming old results to {}.'.format(old_list))
            else:
                logger_save_list.warning('Overwriting old results without saving: {}'.format(file_csv))
                os.remove(file_csv)
        with open(file_csv, 'w+', newline='') as file:
            file.write('TRACK_ID,POSITION_T,POSITION_X,POSITION_Y,WIDTH,HEIGHT,DEGREES_ANGLE\n')  # first row
        return old_list  # return state of old_list, stop function

    if coords is not None:  # Check if we actually received something
        string_holder = ''  # Create empty string to which rows are appended
        for item in coords:
            # convert tuple first into single parts, then to .csv row
            frame, obj_id, xy, (w, h, deg) = item
            x, y = xy[:2]  # in case of (x, y, illumination)
            curr_string = '{0},{1},{2},{3},{4},{5},{6}\n'.format(
                int(obj_id),  # 0  # Appeared sometimes as float; intercepted here
                int(frame),  # 1
                x,  # 2
                y,  # 3
                w,  # 4
                h,  # 5
                deg  # 6
            )
            string_holder += curr_string  # append row
        with open(file_csv, 'a', newline='') as file:  # append rows to .csv file
            file.write(string_holder)


def set_different_colour_filter(colour_filter_new):
    logger = logging.getLogger('ei').getChild(__name__)
    logger.warning('Setting colour filter to {}'.format(colour_filter_new))
    flags = [flag for flag in dir(cv2) if flag.startswith('COLOR_')]  # get all possible flags
    if colour_filter_new.isdigit():
        colour_filter_new = int(colour_filter_new)
    elif colour_filter_new not in flags:  # check if input is within flags
        logger.critical('Could not find color_filter. Available filters:')
        for i in flags:
            logger.critical('{}'.format(i))
        logger.critical('Please update tracking.ini accordingly.')
        sys.exit('Please update tracking.ini accordingly.')
    else:
        try:
            colour_filter_new = eval('cv2.{}'.format(colour_filter_new))  # try to set flag
        except Exception as ex:  # just in case
            template = 'An exception of type {0} occurred during set_different_colour_filter(). Arguments:\n{1!r}'
            logger.exception(template.format(type(ex).__name__, ex.args))
            logger.critical('Could not update color_filter. String provided: {}.\n'
                            'Provided name should have been within possible names. Please update color_filter '
                            'in source code.\nSorry.'.format(colour_filter_new))
            sys.exit('Could not update color_filter. String provided: {}.\n'
                     'Provided name should have been within possible names. Please update color_filter '
                     'in source code.\nSorry.'.format(colour_filter_new))
    return colour_filter_new  # return flag


def shift_np_array(arr, shift, fill_value=np.nan):
    """
    # preallocate empty array and assign slice by chrisaycock
    # See origin:
    # https://stackoverflow.com/questions/30399534/shift-elements-in-a-numpy-array
    # Accessed last 2019-04-24 13:37:00,101
    """
    result = np.empty_like(arr)
    if shift > 0:
        result[:shift] = fill_value
        result[shift:] = arr[:-shift]
    elif shift < 0:
        result[shift:] = fill_value
        result[:shift] = arr[-shift:]
    else:
        result[:] = arr
    return result


def sort_list(file_path=None, sort=None, df=None, save_file=True):
    logger = logging.getLogger('ei').getChild(__name__)
    if sort is None:
        sort = ['TRACK_ID', 'POSITION_T']
    if file_path is not None and df is None:
        df = get_data(file_path)  # get data frame from .csv
        logger.debug('Sorting list {}'.format(file_path))
    if not isinstance(df, pd.core.frame.DataFrame):
        error_msg = 'No/wrong arguments passed to sort_list(): file_path: {}, sort: {}, df: {} save_file: {}'.format(
            file_path, sort, type(df), save_file)
        logger.critical(error_msg)
        return None
    try:
        df.sort_values(by=sort, inplace=True, na_position='first')  # Sort data frame
        df.reset_index(drop=True, inplace=True)  # reset index of df
    except Exception as ex:
        template = 'An exception of type {0} occurred while sorting file {2}. Arguments:\n{1!r}'
        logger.exception(template.format(type(ex).__name__, ex.args, file_path))
        return None
    if save_file and file_path is not None:
        try:
            with open(file_path, 'w+', newline='\n') as csv:  # save again as csv
                df.to_csv(csv, index=False)
            logger.info('Results saved to: {}'.format(file_path))
        except Exception as ex:
            template = 'An exception of type {0} occurred while saving file {2} after sorting. Arguments:\n{1!r}'
            logger.exception(template.format(type(ex).__name__, ex.args, file_path))
            return None
    elif save_file and file_path is None:
        logger.critical('Cannot save file if no file path is provided.')
    return df


if __name__ == '__main__':
    _backup(skip_check_time=False)
    create_configs()
    d = get_configs()
    if d is not None:
        for dic in d:
            print('{}'.format('#' * 50))
            for key in dic:
                print(key, ': ', dic[key])
                if dic[key] is None:
                    print('MISSING: {}'.format(key))
    sys.exit()
