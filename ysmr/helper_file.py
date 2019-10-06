#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Copyright 2019 Julian Schwanbeck (julian.schwanbeck@med.uni-goettingen.de)
https://github.com/schwanbeck/YSMR
##Explanation
This file contains various functions used by YSMR.
This file is part of YSMR. YSMR is free software: you can distribute it and/or modify
it under the terms of the GNU General Public License as published by the Free Software Foundation,
either version 3 of the License, or (at your option) any later version. YSMR is distributed in
the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
details. You should have received a copy of the GNU General Public License along with YSMR. If
not, see <http://www.gnu.org/licenses/>.
"""

import collections
import configparser
import logging
import os
import platform
import subprocess
import sys
from datetime import datetime
from glob import glob
from logging.handlers import QueueHandler, QueueListener
from queue import Queue
from time import localtime, strftime
from tkinter import filedialog, Tk

import cv2
import numpy as np
import pandas as pd
from scipy.signal import argrelextrema

_config = configparser.ConfigParser(allow_no_value=True)


def argrelextrema_groupby(group, comparator=np.greater_equal, order=10, shift_range=4, fill_value=0):
    """
    Find local minima/maxima in range of order (depending on comparator).
    When shift is not 0, will only return one result in range of shift.
    Returns array with non-extrema replaced by fil_value

    :param group: array like
    :param comparator: numpy comparator or equivalent
    :param order: range in which to check for extrema
    :type order: int
    :param shift_range: range in which to exclude multiple results (0 is off).
    :type shift_range: int
    :param fill_value: fill value for array
    :return: group minima/maxima
    :rtype: pd.Series
    """
    group_intermediate = group.to_numpy()
    result = np.zeros(group.shape[0], dtype=np.int8)
    np.put(result, argrelextrema(group_intermediate, comparator, order=order)[0], 1)
    if shift_range:
        result_comp = result
        for d_shift in range(-1, -(shift_range + 1)):
            query = shift_np_array(result_comp, d_shift, 0)
            result = np.where((
                (result == 1) &
                (query == 1),
                0, result))
    result = np.where(result == 1, group_intermediate, fill_value)
    result = pd.Series(result, index=group.index)
    return result


def bytes_to_human_readable(number_of_bytes):
    """
    Inspired by https://stackoverflow.com/questions/44096473/
    Returns string containing bytes, rounded to 1 decimal place,
    with unit prefix as defined by SI.
    For documentation purposes only.

    :param number_of_bytes: bytes to convert
    :type number_of_bytes: int
    :return: the readable number as text
    :rtype: str
    """
    if number_of_bytes < 0:  # assert number_of_bytes >= 0, 'Negative bytes passed'
        return 'Negative Bytes'  # As this isn't / shouldn't be used anywhere important, this suffices
    bytes_in_a_kb = 1024  # 2**10, as it should be
    units = ['bytes', 'KB', 'MB', 'GB', 'TB', 'PB', 'EB', 'ZB', 'YB', ]  # Update as needed
    for unit in units:  # Go through each unit
        if number_of_bytes / bytes_in_a_kb < 1 or unit == units[-1]:
            break  # stop if division would go below 1 or end of list is reached
        number_of_bytes /= bytes_in_a_kb  # divide otherwise
    return '{0:.01f} {1}'.format(number_of_bytes, unit)


def collate_results_csv_to_xlsx(path=None, save_path=None, csv_extension='statistics.csv'):
    """
    Saves all available csv files ending in the specified csv_extension into one xlsx

    :param path: folder with csv files
    :param save_path: output folder
    :param csv_extension: extension of .csv files
    :return: None
    """
    logger = logging.getLogger('ysmr').getChild(__name__)
    try:
        import xlsxwriter
    except ImportError:
        logger.warning('Could not import module \'xlsxwriter\', saving as .xlsx file is not possible.')
        return
    if save_path is None:
        save_path = './'
    if path is None:
        path = get_any_paths(rename=False, file_types=[
            ('csv', '.csv'),
            ('all files', '.*'),
        ])
    file_path = '{}{}_collated_statistics.xlsx'.format(save_path, datetime.now().strftime('%y%m%d%H%M%S'))
    paths = find_paths(base_path=path, extension=csv_extension)
    if paths:
        writer = pd.ExcelWriter(file_path, engine='xlsxwriter')
        for path in paths:
            with open(path, 'r', newline='\n') as csv:
                df = pd.read_csv(csv,
                                 # chunksize=10 ** 6,
                                 sep=',',
                                 header=0,
                                 # usecols=[],
                                 # dtype=[],
                                 # names=[],
                                 encoding='utf-8',
                                 )
            file_name = os.path.basename(path)
            file_name = os.path.splitext(file_name)[0]
            # Limit max rows and sheet name length
            df.loc[:2**20 - 1, :].to_excel(writer, sheet_name=file_name[:31])
        writer.save()
        logger.info('Collated results: {}'.format(os.path.abspath(file_path)))
    else:
        logger.info('Could not find paths.')


def create_configs(config_filepath=None):
    """
    generates the tracking.ini config file and tries to open it for editing.

    :param config_filepath: optional file path
    :return: None
    """
    logger = logging.getLogger('ysmr').getChild(__name__)
    # path = os.path.join(os.path.expanduser("~"), '.config/ysmr')
    # import xdg -> ~/$XDG_CONFIG_HOME/ysmr
    # %APPDATA%: path = os.getenv('APPDATA') + ysmr
    if config_filepath is None:
        config_filepath = os.path.join(os.path.abspath('./'), 'tracking.ini')
    try:
        config_path_root, config_file_ext = os.path.splitext(config_filepath)
        old_tracking_ini = '{}_{}{}'.format(config_path_root, datetime.now().strftime('%y%m%d%H%M%S'), config_file_ext)
        os.rename(config_filepath, old_tracking_ini)
        logger.warning('Old tracking.ini renamed to {}'.format(old_tracking_ini))
    except FileNotFoundError:
        pass

    _config['BASIC RECORDING SETTINGS'] = {
        'pixel per micrometre': 1.41888781,
        'frames per second': 30.0,
        'frame height': 922,
        'frame width': 1228,
        'white bacteria on dark background': True,
        'rod shaped bacteria': True,
        'threshold offset for detection': 5,
    }

    _config['BASIC TRACK DATA ANALYSIS SETTINGS'] = {
        'minimal length in seconds': 20.0,
        'limit track length to x seconds': 20.0,
        'minimal angle in degrees for turning point': 30.0,
        'extreme area outliers lower end in px*px': 2,
        'extreme area outliers upper end in px*px': 50,
    }

    _config['DISPLAY SETTINGS'] = {
        'user input': True,
        'select files': True,
        'display video analysis': True,
        'save video': False,
    }

    _config['RESULTS SETTINGS'] = {
        'rename previous result .csv': False,
        'delete .csv file after analysis': False,
        'store processed .csv file': True,
        'store generated statistical .csv file': True,
        'store final analysed .csv file': True,
        'split results by (Turn Points / Distance / Speed / Time / Displacement / perc. motile)': 'perc. motile',
        'split violin plots on': '0.0, 20.0, 40.0, 60.0, 80.0, 100.0',
        'save large plots': True,
        'save rose plot': True,
        'save time violin plot': True,
        'save acr violin plot': True,
        'save length violin plot': True,
        'save turning point violin plot': True,
        'save speed violin plot': True,
        'save angle distribution plot / bins': 36,
        'save displacement violin plot': True,
        'save percent motile plot': True,
        'collate results csv to xlsx': True,
    }

    _config['LOGGING SETTINGS'] = {
        'log to file': True,
        'log file path': './logfile.log',
        'shorten displayed logging output': False,
        'shorten logfile logging output': False,
        'set logging level (debug/info/warning/critical)': 'debug',
        'verbose': False,
    }

    _config['ADVANCED VIDEO SETTINGS'] = {
        'include luminosity in tracking calculation': False,
        'color filter': 'COLOR_BGR2GRAY',
        'minimal frame count': 600,
        'stop evaluation on error': True,
        'list save length interval': 10000,
        'save video file extension': '.mp4',
        'save video fourcc codec': 'mp4v',
    }

    _config['ADVANCED TRACK DATA ANALYSIS SETTINGS'] = {
        'maximal consecutive holes': 5,
        'maximal empty frames in %': 5.0,
        'percent quantiles excluded area': 10.0,
        'try to omit motility outliers': True,
        'stop excluding motility outliers if total count above percent': 5.0,
        'exclude measurement when above x times average area': 1.5,
        'rod average width/height ratio min.': 0.125,
        'rod average width/height ratio max.': 0.67,
        'coccoid average width/height ratio min.': 0.8,
        'coccoid average width/height ratio max.': 1.0,
        'percent of screen edges to exclude': 5.0,
        'maximal recursion depth': 960,
        'limit track length exactly': False,
        'compare angle between n frames': 10,
        'force tracking.ini fps settings': False,
    }

    _config['HOUSEKEEPING'] = {
        'previous directory': './',
        'shut down after analysis': False,
    }

    _config['TEST SETTINGS'] = {
        'debugging': False,
        'path to test video': 'Q:/test_video.avi',
    }

    try:
        with open(config_filepath, 'w+') as configfile:
            _config.write(configfile)
        logger.critical('tracking.ini was reset to default values. Path: {}'.format(config_filepath))
    except (IOError, OSError) as configfile_error:
        logger.exception('Could not create config file: {}'.format(configfile_error))
        return
    try:
        """
        Opens text file after creation, should be correct for win/linux/mac
        Source: Ch.Idea
        https://stackoverflow.com/questions/434597/
        Accessed last 2019-06-07 13:37:00,101        
        """
        # @todo: untested on mac
        if os.name is 'nt':  # Windows
            # works with spaces in name whereas subprocess.call(('start', path), shell=True) does not
            response = subprocess.run(('cmd /c start "" "{}"'.format(config_filepath)), stderr=subprocess.PIPE)
        elif sys.platform.startswith('darwin'):  # Mac
            response = subprocess.call(('open', config_filepath), stderr=subprocess.PIPE)
        else:  # Linux
            response = subprocess.call(('xdg-open', config_filepath), stderr=subprocess.PIPE)
        response.check_returncode()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError) as file_open_error:
        logger.exception(file_open_error)
    finally:
        # As the file has to be checked first and the process could
        # continue with a freshly generated one, we'll stop execution here.
        logger.critical('Created new tracking.ini. Please check the values in the file: {}'.format(config_filepath))
        sys.exit('Created new tracking.ini. Please check the values in the file: {}'.format(config_filepath))


# Check tracking.ini / create it --> moved into functions
# TRACKING_INI_FILEPATH = os.path.join(os.path.abspath('./'), 'tracking.ini')
# if not os.path.isfile(TRACKING_INI_FILEPATH):  # needed by later functions
#     create_configs(config_filepath=TRACKING_INI_FILEPATH)
# _config.read(TRACKING_INI_FILEPATH)


def check_logfile(path, max_size=2 ** 20):  # max_size=1 MB
    """
    Checks if logfile is above specified size and does a rollover if necessary
    If not, checks if file is padded with empty lines and adds some if necessary

    :param path: path to logfile
    :param max_size: maximal size of logfile in bytes
    :type max_size: int
    :return: log-file path
    :rtype: str
    """
    # RotatingFileHandler does the same but can't do the
    # rollover when used with multiprocess, so we create our own approximation
    if os.path.isfile(path):
        file_size = os.path.getsize(path)
    else:
        file_size = 0
    if file_size < max_size:
        if 0 < file_size:  # if the file already contains lines, check if we start on an empty line
            logfile_padding(path)  # or pad it with empty lines
    else:
        base_path, file_name = os.path.split(path)
        old_paths = find_paths(base_path=base_path, extension='{}.*'.format(file_name), recursive=False)
        if old_paths:  # rename old files from .log.1 to .log.9; delete otherwise
            old_paths = sorted(old_paths, reverse=True, key=lambda x: int(x[-1]))
            counts = [int(count[-1]) for count in old_paths]
            if not counts[-1] > 1:  # if smallest number isn't 1, we can stop
                max_idx = [1]
                max_idx.extend([s - t for s, t in zip(counts[:-1], counts[1:])])
                max_idx = np.array(max_idx).argmax()  # look for a gap in the numbering
                for old_count, old_path in zip(counts[max_idx:], old_paths[max_idx:]):
                    try:
                        if old_count == 9:
                            os.remove(old_path)
                        else:
                            new_path = '{}{}'.format(old_path[:-1], old_count + 1)
                            if not os.path.isfile(new_path):
                                os.rename(old_path, new_path)
                    except (FileNotFoundError, FileExistsError, PermissionError):
                        pass
        try:
            os.rename(path, '{}.1'.format(path))
        except (FileNotFoundError, FileExistsError, PermissionError):
            pass
    return path


def create_results_folder(path):
    """
    creates a dated result folder in provided path

    :param path: path to folder
    :return: path to result folder
    """
    logger = logging.getLogger('ysmr').getChild(__name__)
    dir_form = '{}_Results/'.format(str(strftime('%y%m%d', localtime())))
    if isinstance(path, str) or isinstance(path, os.PathLike):
        pass
    elif isinstance(path, list) or isinstance(path, tuple):
        path = path[0]
    else:
        path = './'
        logger.critical('Could not access base path in path to files; '
                        'results folder created in {}'.format(os.path.abspath(path)))
    directory = os.path.abspath(os.path.join(os.path.dirname(path), dir_form))
    if not os.path.exists(directory):
        try:
            make_dir(directory)
            logger.info('Results folder: {}'.format(directory))
        except OSError as makedir_error:
            logger.exception(makedir_error)
            logger.warning('Unable to create {}, Directory changed to {}'.format(
                directory, os.path.abspath('./')))
            directory = './'
        finally:
            pass
    return directory


def creation_date(path_to_file):
    """
    Try to get the date that a file was created, falling back to when it was
    last modified if that isn't possible.
    See for explanation/source:
    https://stackoverflow.com/a/39501288/1709587
    Accessed last 2019-03-01 13:37:00,101

    :param path_to_file: path to file
    :return: time since file creation in seconds
    :rtype: int
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
    """
    check for changes in column, return lists of starts/stops

    :param data: pandas data frame
    :param column: column to be checked for changes
    :return: list of start IDs, list of stop IDs
    :rtype starts: list
    :rtype stops: list
    """
    track_id = data[column].array
    # get np.ndarray with the values where track changes occur
    # by comparing every element with the previous one
    stops = np.where(track_id[:-1] != track_id[1:])[0]
    stops = stops.tolist()  # convert to list for manipulation
    starts = [0]  # Initialise starts; track 0 starts at index 0 obviously
    starts.extend([item + 1 for item in stops])  # others start at stop + 1
    stops.append(data.index.max())  # Append last stop to list of stops
    return starts, stops


def elapsed_time(time_one):
    """
    rough measure for elapsed time since time_one

    :param time_one: start time
    :return: time difference
    """
    logger = logging.getLogger('ysmr').getChild(__name__)
    time_two = datetime.now()
    try:
        time_delta = time_two - time_one
    except ValueError as val_error:
        logger.exception(val_error)
        return None
    return time_delta


def find_paths(base_path, extension, minimal_age=0, maximal_age=np.inf, recursive=True):
    """
    Search for files with provided extension in provided path

    :param base_path: path which is checked for files
    :param extension: extension or ending of files
    :type extension: str
    :param minimal_age: minimal file age in seconds
    :type minimal_age: int
    :param maximal_age: maximal file age in seconds
    :param recursive: whether to check sub-folders
    :type recursive: bool
    :return: list of files
    """
    logger = logging.getLogger('ysmr').getChild(__name__)
    if not os.path.exists(base_path):
        logger.warning('Path could not be found: {}'.format(base_path))
        return None
    if base_path[-1] != '/':
        base_path = '{}/'.format(base_path)

    base_path = '{}**/*{}'.format(base_path, extension)
    in_files = glob(base_path, recursive=recursive)
    out_files = []
    for file in in_files:
        # Set path separators to /
        file = file.replace(os.sep, '/')
        # checks if file is young/old enough
        file_creation_date = creation_date(file)
        if file_creation_date >= 0 or (file_creation_date < 0 and minimal_age < 0):
            if maximal_age >= file_creation_date >= minimal_age:
                out_files.append(file)
            elif file_creation_date < 0 and minimal_age < 0:
                out_files.append(file)
        else:
            logger.warning('The file appears to be {:.2f} seconds '
                           'from the future and was thus not selected. '
                           'File: {}'.format(abs(file_creation_date), file))
    return out_files


def get_any_paths(prev_dir=None, rename=False, file_types=None, settings=None):
    """
    Ask user for file selection with a tkinter askopenfilenames.

    :param prev_dir: Folder in which to start search
    :param rename: Whether to rename the previous folder in the config file tracking.ini
    :type rename: bool
    :param file_types: Optional list of file extensions which can be included, first extension will be used as default
    :param settings: tracking.ini settings
    :return: list of files
    :rtype: list
    """
    logger = logging.getLogger('ysmr').getChild(__name__)
    settings = get_configs(settings)
    _config.read(settings['tracking_ini_filepath'])
    if prev_dir is None:
        try:
            prev_dir = _config['HOUSEKEEPING'].get('previous directory', fallback='./')
        except configparser.Error:
            prev_dir = './'
    if file_types is None:
        file_types = [
            ('all files', '.*'),
            ('csv', '.csv'),
            ('avi', '.avi'),
            ('mkv', '.mkv'),
            ('mov', '.mov'),
            ('mp4', '.mp4'),
        ]
    try:
        root = Tk()
        root.overrideredirect(1)  # hide the root window
        root.withdraw()
        paths = filedialog.askopenfilenames(
            title='Choose files. ',
            filetypes=file_types,
            defaultextension=file_types[0][1],
            multiple=True,
            initialdir=prev_dir,
        )
    except Exception as ex:
        template = 'An exception of type {0} occurred. Arguments:\n{1!r}'
        logger.exception(template.format(type(ex).__name__, ex.args))
        return None
    if paths and rename:
        curr_path = os.path.dirname(paths[0])
        try:
            _config.set('HOUSEKEEPING', 'previous directory', curr_path)
            with open(settings['tracking_ini_filepath'], 'w') as configfile:
                _config.write(configfile)
            logger.debug('Previous directory set to {}'.format(curr_path))
        except (PermissionError, configparser.Error, FileNotFoundError):
            pass
        except Exception as ex:
            template = 'An exception of type {0} occurred. Arguments:\n{1!r}'
            logger.exception(template.format(type(ex).__name__, ex.args))
    return paths


def get_configs(tracking_ini_filepath=None):
    """
    Read tracking.ini, convert values to usable form and return as dict

    :param tracking_ini_filepath: filepath for tracking.ini
    :return: configs as dict
    :rtype: dict
    """
    logger = logging.getLogger('ysmr').getChild(__name__)
    settings_dict = None
    if isinstance(tracking_ini_filepath, collections.Mapping):  # lazy check for already generated settings
        settings_dict = tracking_ini_filepath
    else:
        # if tracking_ini_filepath is not None and os.path.isfile(tracking_ini_filepath):
        #     _config.read(tracking_ini_filepath)
        if tracking_ini_filepath is None:
            # tracking_ini_filepath = TRACKING_INI_FILEPATH
            tracking_ini_filepath = os.path.join(os.path.abspath('./'), 'tracking.ini')
        tracking_ini_filepath = os.path.abspath(tracking_ini_filepath)
        _config.read(tracking_ini_filepath)
        try:
            # try to get all configs
            basic_recording = _config['BASIC RECORDING SETTINGS']
            basic_track = _config['BASIC TRACK DATA ANALYSIS SETTINGS']
            display = _config['DISPLAY SETTINGS']
            results = _config['RESULTS SETTINGS']
            log_settings = _config['LOGGING SETTINGS']
            adv_video = _config['ADVANCED VIDEO SETTINGS']
            adv_track = _config['ADVANCED TRACK DATA ANALYSIS SETTINGS']
            housekeeping = _config['HOUSEKEEPING']
            test = _config['TEST SETTINGS']

            # Convert some values directly into usable form
            verbose = log_settings.getboolean('verbose')
            set_log_level = log_settings.get('set logging level (debug/info/warning/critical)')
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
            rod_shaped_bac = basic_recording.getboolean('rod shaped bacteria')
            if rod_shaped_bac:
                min_size_ratio = adv_track.getfloat('rod average width/height ratio min.')
                max_size_ratio = adv_track.getfloat('rod average width/height ratio max.')
            else:
                min_size_ratio = adv_track.getfloat('coccoid average width/height ratio min.')
                max_size_ratio = adv_track.getfloat('coccoid average width/height ratio max.')
            colour_filter = adv_video.get('color filter')
            if colour_filter != 'COLOR_BGR2GRAY':
                colour_filter = set_different_colour_filter(colour_filter)
            else:
                colour_filter = cv2.COLOR_BGR2GRAY
            # convert string to list of floats
            split_on_percentage = [float(i.strip()) for i in results.get('split violin plots on').split(',')]

            # one large dict so we can pass it around between functions
            settings_dict = {
                # _config['BASIC RECORDING SETTINGS']
                'pixel per micrometre': basic_recording.getfloat('pixel per micrometre'),
                'frames per second': basic_recording.getfloat('frames per second'),
                'frame height': basic_recording.getint('frame height'),
                'frame width': basic_recording.getint('frame width'),
                'white bacteria on dark background': basic_recording.getboolean(
                    'white bacteria on dark background'),
                'rod shaped bacteria': basic_recording.getboolean('rod shaped bacteria'),
                'threshold offset for detection': basic_recording.getint('threshold offset for detection'),

                # _config['BASIC TRACK DATA ANALYSIS SETTINGS']
                'minimal length in seconds': basic_track.getfloat('minimal length in seconds'),
                'limit track length to x seconds': basic_track.getfloat('limit track length to x seconds'),
                'minimal angle in degrees for turning point': basic_track.getfloat(
                    'minimal angle in degrees for turning point'),
                'extreme area outliers lower end in px*px': basic_track.getint(
                    'extreme area outliers lower end in px*px'),
                'extreme area outliers upper end in px*px': basic_track.getint(
                    'extreme area outliers upper end in px*px'),

                # _config['DISPLAY SETTINGS']
                'user input': display.getboolean('user input'),
                'select files': display.getboolean('select files'),
                'display video analysis': display.getboolean('display video analysis'),
                'save video': display.getboolean('save video'),

                # _config['RESULTS SETTINGS']
                'rename previous result .csv': results.getboolean('rename previous result .csv'),
                'delete .csv file after analysis': results.getboolean('delete .csv file after analysis'),
                'store processed .csv file': results.getboolean('store processed .csv file'),
                'store generated statistical .csv file': results.getboolean(
                    'store generated statistical .csv file'),
                'store final analysed .csv file': results.getboolean('store final analysed .csv file'),
                'split results by (Turn Points / Distance / Speed / Time / Displacement / perc. motile)':
                    results.get(
                        'split results by (Turn Points / Distance / Speed / Time / Displacement / perc. motile)'
                    ),
                'split violin plots on': split_on_percentage,
                'save large plots': results.getboolean('save large plots'),
                'save rose plot': results.getboolean('save rose plot'),
                'save time violin plot': results.getboolean('save time violin plot'),
                'save acr violin plot': results.getboolean('save acr violin plot'),
                'save length violin plot': results.getboolean('save length violin plot'),
                'save turning point violin plot': results.getboolean('save turning point violin plot'),
                'save speed violin plot': results.getboolean('save speed violin plot'),
                'save angle distribution plot / bins': results.getint('save angle distribution plot / bins'),
                'save displacement violin plot': results.getboolean('save displacement violin plot'),
                'save percent motile plot': results.getboolean('save percent motile plot'),
                'collate results csv to xlsx': results.getboolean('collate results csv to xlsx'),

                # _config['LOGGING SETTINGS']
                'log to file': log_settings.getboolean('log to file'),
                'log file path': log_settings.get('log file path'),
                'shorten displayed logging output': log_settings.getboolean(
                    'shorten displayed logging output'),
                'shorten logfile logging output': log_settings.getboolean(
                    'shorten logfile logging output'),
                'set logging level (debug/info/warning/critical)': set_log_level,
                'log_level': set_log_level_setting,
                'verbose': verbose,

                # _config['ADVANCED VIDEO SETTINGS']
                'include luminosity in tracking calculation': adv_video.getboolean(
                    'include luminosity in tracking calculation'),
                'color filter': colour_filter,
                'minimal frame count': adv_video.getint('minimal frame count'),
                'stop evaluation on error': adv_video.getboolean('stop evaluation on error'),
                'list save length interval': adv_video.getint('list save length interval'),
                'save video file extension': adv_video.get('save video file extension'),
                'save video fourcc codec': adv_video.get('save video fourcc codec'),

                # _config['ADVANCED TRACK DATA ANALYSIS SETTINGS']
                'maximal consecutive holes': adv_track.getint('maximal consecutive holes'),
                'maximal empty frames in %': adv_track.getfloat('maximal empty frames in %') / 100 + 1,
                'percent quantiles excluded area': adv_track.getfloat('percent quantiles excluded area') / 100,
                'try to omit motility outliers': adv_track.getboolean('try to omit motility outliers'),
                'stop excluding motility outliers if total count above percent': adv_track.getfloat(
                    'stop excluding motility outliers if total count above percent') / 100,
                'exclude measurement when above x times average area': adv_track.getfloat(
                    'exclude measurement when above x times average area'),
                'average width/height ratio min.': min_size_ratio,
                'average width/height ratio max.': max_size_ratio,
                'percent of screen edges to exclude': adv_track.getfloat('percent of screen edges to exclude') / 100,
                'maximal recursion depth': adv_track.getint('maximal recursion depth'),  # 0 off
                'limit track length exactly': adv_track.getboolean('limit track length exactly'),
                'compare angle between n frames': adv_track.getint('compare angle between n frames'),
                'force tracking.ini fps settings': adv_track.getboolean('force tracking.ini fps settings'),

                # _config['HOUSEKEEPING']
                'previous directory': housekeeping.get('previous directory', fallback='./'),
                'shut down after analysis': housekeeping.getboolean('shut down after analysis'),

                # _config['TEST SETTINGS']
                'debugging': test.getboolean('debugging'),
                'path to test video': test.get('path to test video'),

                # Internal
                'tracking_ini_filepath': tracking_ini_filepath,
            }
            if verbose:
                logger.debug('tracking.ini settings:')
            for test_key in settings_dict:
                if settings_dict[test_key] is None:
                    error = 'tracking.ini is missing a value in {}'.format(test_key)
                    logger.critical(error)
                    settings_dict = None
                    break
                elif verbose:
                    logger.debug('{}: {}'.format(test_key, settings_dict[test_key]))
        except (TypeError, ValueError, KeyError) as ex:  # Exception
            template = 'An exception of type {0} occurred while attempting to read tracking.ini. Arguments:\n{1!r}'
            logger.exception(template.format(type(ex).__name__, ex.args))

    if not settings_dict:  # something went wrong, presumably missing/broken entries or sections
        create_configs(config_filepath=tracking_ini_filepath)  # re-create tracking.ini
        return None
    return settings_dict


def get_data(csv_file_path, dtype=None, check_sorted=True):
    """
    load csv file to pandas data frame

    :param csv_file_path: csv file to read
    :param dtype: dict of columns to be loaded and their data types
    :type dtype: dict
    :param check_sorted: check if df is sorted by TRACK_ID / POSITION_T if available
    :type check_sorted: bool
    :return: pandas data frame
    """
    logger = logging.getLogger('ysmr').getChild(__name__)
    if type(csv_file_path) is not (str or os.PathLike or bytes) and (list or tuple):
        csv_file_path = csv_file_path[0]
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
                 'DEGREES_ANGLE': np.float64
                 }
    use_cols = list(dtype.keys())
    try:
        # Fixes some special character problems with pd.read_csv paths:
        with open(csv_file_path, 'r', newline='\n') as csv:
            # csv_chunks =  # use chunks in case file is too large
            # Done automatically by pd.read_csv()
            df = pd.read_csv(csv,
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
    # rough check if file is sorted
    if all(x in use_cols for x in ['TRACK_ID', 'POSITION_T']) and check_sorted:
        series_check_sorted = df.loc[:5, 'TRACK_ID']
        if series_check_sorted.is_unique:
            logger.info('The data frame seems not to be sorted by \'TRACK_ID\' and \'POSITION_T\', sorting now.')
            df = sort_list(df=df, save_file=False)
    logger.debug('Done reading {} into data frame'.format(csv_file_path))
    return df


def get_loggers(log_level=logging.DEBUG, logfile_name='./logfile.log',
                short_stream_output=False, short_file_output=False, log_to_file=False):
    """
    looks if loggers are already set up, creates new ones if they are missing.
    Workaround for multiprocessing logging.

    :param log_level: minimal logging level
    :param logfile_name: file for logging.Filehandler
    :param short_stream_output: shorten the sys.stdout output
    :type short_file_output: bool
    :param short_file_output: shorten the logging.Filehandler output
    :type short_file_output: bool
    :param log_to_file: whether to use a logfile
    :type log_to_file: bool
    :return: logging queue, longest used logging format
    """
    logger = logging.getLogger('ysmr')
    logger.propagate = False
    # Log message setup
    long_format_logging = '{asctime:}\t' \
                          '{funcName:15.15}\t' \
                          '{lineno:>4}\t' \
                          '{levelname:8.8}\t' \
                          '{process:>5}:\t' \
                          '{message}'
    # '{name:19.19}\t' \  # logger name
    # '{filename:18:18}\t' \  # file name
    short_format_logging = '{asctime:}\t' \
                           '{levelname:8.8}\t' \
                           '{process:>5}:\t' \
                           '{message}'
    # Sets the global logging format.
    logging.basicConfig(format=(long_format_logging, ), style='{')  # ISO8601: "%Y-%m-%dT%H:%M:%S%z"
    queue_listener = None
    if len(logger.handlers) > 0:
        for handler in logger.handlers:
            if isinstance(handler, QueueListener):
                queue_listener = handler  # if we have our handler, we can stop
                break
    if not queue_listener:  # otherwise, we have to set it up
        logger_formatter = logging.Formatter(long_format_logging, style='{')  # ISO8601: , "%Y-%m-%dT%H:%M:%S%z"
        short_logger_formatter = logging.Formatter(short_format_logging, style='{')
        logger.setLevel(log_level)
        log_queue = Queue(-1)
        queue_handler = QueueHandler(log_queue)
        logger.addHandler(queue_handler)
        # Stream handler
        stream_handler_logger = logging.StreamHandler(sys.stdout)
        stream_handler_logger.setLevel(log_level)
        if short_stream_output:
            stream_handler_logger.setFormatter(short_logger_formatter)
        else:
            stream_handler_logger.setFormatter(logger_formatter)
        # File handler
        if log_to_file:
            file_handler_logger = logging.FileHandler(filename=logfile_name, mode='a', )
            file_handler_logger.setLevel(log_level)
            if short_file_output:
                file_handler_logger.setFormatter(short_logger_formatter)
            else:
                file_handler_logger.setFormatter(logger_formatter)
            queue_listener = QueueListener(log_queue, stream_handler_logger, file_handler_logger)
        else:
            queue_listener = QueueListener(log_queue, stream_handler_logger)
        queue_listener.start()

    # If a file is present the file format trumps stream format as the file will be a permanent record
    # and I can't figure out how to log different messages per logger in a way that works consistently
    return_format = long_format_logging
    if log_to_file and short_file_output:
        return_format = short_format_logging
    elif not log_to_file and short_stream_output:
        return_format = short_format_logging
    return queue_listener, return_format


def log_infos(settings, format_for_logging=None):
    """Logging output for several options set in settings.

    :param settings: settings dict from get_configs()
    :type settings: dict
    :param format_for_logging: logging format string
    :type format_for_logging: str
    :return: filler for logger
    :rtype: str
    """
    logger = logging.getLogger('ysmr').getChild(__name__)
    # Log some general stuff
    if format_for_logging is None:
        format_for_logging = '{asctime:}\t' \
                             '{name:21.21}\t' \
                             '{funcName:14.14}\t' \
                             '{lineno:>4}\t' \
                             '{levelname:8.8}\t' \
                             '{process:>5}:\t' \
                             '{message}'
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
        logger.warning('Running cv2 unoptimised; check openCV documentation for help.')
    if not settings['rename previous result .csv']:
        logger.warning('Old .csv result lists will be overwritten')
    if settings['delete .csv file after analysis']:
        logger.warning('Generated .csv files will be deleted after analysis')
    if settings['select files'] and settings['debugging']:
        logger.warning('Manually selecting files disabled due to debugging')

    # Infos
    logger.info('Settings file location: {}'.format(os.path.abspath(settings['tracking_ini_filepath'])))
    logger.info('Logfile location: {}'.format(os.path.abspath(settings['log file path'])))
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
        limit_track_string = 'Maximal track length for evaluation set to {} s'.format(
            settings['limit track length to x seconds'])
        if settings['limit track length exactly']:
            limit_track_string += ' exactly. Tracks off by any frames will be discarded.'
        logger.info(limit_track_string)
    else:
        logger.info('Full track length will be used in evaluation')
    if not settings['maximal recursion depth']:
        logger.info('Tracks will not be split on error as \'maximal recursion depth\' is set to 0. '
                    'This could severely reduce the number of viable tracks.')

    # Debug messages
    logger.debug('White bacteria on dark background set to {}'.format(
        settings['white bacteria on dark background']))
    logger.debug('List save length set to {} entries'.format(settings['list save length interval']))
    logger.debug('Pixel/micrometre: {}'.format(settings['pixel per micrometre']))
    if settings['verbose']:
        logger.debug('tracking.ini settings:')
        for key in settings:
            logger.debug('{}: {}'.format(key, settings[key]))
    return filler_for_logger


def logfile_padding(logfile, iteration=0):
    """pads text file with max. two empty lines if it doesn't have one at the end

    :param logfile: path to file
    :param iteration: internal iteration counter
    :return: None
    """
    with open(logfile, 'r+') as file:
        for line in file:
            pass  # get to last line
        if line:  # in case of empty file
            if line not in {'\n', '\r', '\r\n'}:  # check if line contains a newline character
                file.write('\n')  # python substitutes \n for OS specific one
            else:
                return
        else:
            return
    if iteration < 2:  # so we don't accidentally fill the file with empty lines if anything goes wrong
        logfile_padding(logfile, iteration=iteration + 1)


def make_dir(new_directory):
    """works the way a good mkdir should :)
        - already exists, silently complete
        - regular file in the way, raise an exception
        - parent directory(ies) does not exist, make them as well
        Source:
        Originally published: 2001-10-18 10:53:13
        Last updated: 2009-12-18 15:33:58
        Author: Trent Mick
        - https://code.activestate.com/recipes/82465-a-friendly-mkdir/
        Accessed last 2019-06-04 13:37:00,101

    :param new_directory: file path to be created
    :return: None
    """
    if os.path.isdir(new_directory):
        pass
    elif os.path.isfile(new_directory):
        raise OSError('A file with the same name as the desired dir, '
                      '\'{}\', already exists.'.format(new_directory))
    else:
        head, tail = os.path.split(new_directory)
        if head and not os.path.isdir(head):
            make_dir(head)
        if tail:
            os.mkdir(new_directory)


def reshape_result(tuple_of_tuples, *args):
    """
    reshape tuple of tuples into (x, y, *args) and (width, height, degrees_orientation)

    :param tuple_of_tuples: (x, y), (w, h), degrees_orientation
    :param args: additional parameters are added to coordinates
    :return: (x, y, *args), (width, height, degrees_orientation)
    """
    (x, y), (w, h), degrees_orientation = tuple_of_tuples  # ((x, y), (w, h), additional_info), xy is centroid
    additional_info = (w, h, degrees_orientation)
    coordinates = [x, y]
    coordinates.extend(args)
    return tuple(coordinates), additional_info


def save_df_to_csv(df, save_path, rename_old_file=True):
    """
    save data frame to csv file

    :param df: pandas data frame
    :param save_path: path to csv file
    :param rename_old_file: whether to try to rename existing files with same name
    :type rename_old_file: bool
    :return: None
    """
    logger = logging.getLogger('ysmr').getChild(__name__)
    if rename_old_file:
        try:
            old_df_path, old_df_ext = os.path.split(save_path)
            old_csv = os.path.join(old_df_path, '{}.{}'.format(
                datetime.now().strftime('%y%m%d%H%M%S'), old_df_ext
            ))
            os.rename(save_path, old_csv)
            logger.critical('Old {} renamed to {}'.format(os.path.basename(save_path), old_csv))
        except (FileNotFoundError, FileExistsError):
            pass
        except Exception as ex:
            template = 'An exception of type {0} occurred while saving ' \
                       'previous file {2} after sorting. Arguments:\n{1!r}'
            logger.exception(template.format(type(ex).__name__, ex.args, save_path))
        finally:
            pass
    try:
        with open(save_path, 'w+', newline='\n') as csv:  # save as csv
            df.to_csv(csv, index=False, encoding='utf-8')
        logger.info('Selected results saved to: {}'.format(save_path))
    except Exception as ex:
        template = 'An exception of type {0} occurred while saving file {2} after sorting. Arguments:\n{1!r}'
        logger.exception(template.format(type(ex).__name__, ex.args, save_path))
    finally:
        pass


def save_list(path, result_folder=None, coords=None, first_call=False, rename_old_list=True, illumination=False):
    """
    Create csv file for results from track_bacteria(), append results

    :param path: path to video file for first call, path to .csv file otherwise
    :param result_folder: optional path to result folder
    :param coords: list of coordinate tuples
    :type coords: list
    :param first_call: whether to create .csv with header
    :type first_call: bool
    :param rename_old_list: whether to rename old .csv with identical name, if it exists
    :type rename_old_list: bool
    :param illumination: whether to save illumination to .csv
    :type illumination: bool
    :return: first_call returns old_list string if it existed and .csv file path
    """
    logger = logging.getLogger('ysmr').getChild(__name__)
    if first_call:  # set up .csv file
        pathname_file, filename_ext = os.path.split(path)
        if result_folder is None:
            pathname = pathname_file
        else:
            pathname = result_folder
        filename = os.path.splitext(filename_ext)[0]
        file_csv = os.path.join(pathname, '{}_list.csv'.format(filename))
        now = datetime.now().strftime('%y%m%d%H%M%S')
        old_list = False
        permission_error = False
        if os.path.isfile(file_csv):
            if rename_old_list:
                old_filename, old_file_extension = os.path.splitext(file_csv)
                old_list = '{}_{}{}'.format(old_filename, now, old_file_extension)
                try:
                    os.rename(file_csv, old_list)  # rename file
                    logger.info('Renaming old results to {}.'.format(old_list))
                except PermissionError:
                    permission_error = True
            else:
                try:
                    os.remove(file_csv)
                    logger.warning('Overwriting old results without saving: {}'.format(file_csv))
                except PermissionError:
                    permission_error = True
        if permission_error:
            old_list = file_csv
            file_csv = '{}/{}_{}_list.csv'.format(pathname, now, filename)
            logger.warning('Permission to change old csv denied, renamed new one to {}'.format(file_csv))
        with open(file_csv, 'w+', newline='') as file:
            if not illumination:
                file.write('TRACK_ID,POSITION_T,POSITION_X,POSITION_Y,WIDTH,HEIGHT,DEGREES_ANGLE\n')  # first row
            else:
                file.write('TRACK_ID,POSITION_T,POSITION_X,POSITION_Y,WIDTH,HEIGHT,DEGREES_ANGLE,ILLUMINATION\n')
        return old_list, file_csv

    if coords:  # Check if we actually received something
        string_holder = ''  # Create empty string to which rows are appended
        for item in coords:
            # convert tuple first into single parts, then to .csv row
            frame, obj_id, xy, (w, h, deg) = item
            x, y = xy[:2]  # in case of (x, y, illumination)
            curr_string = '{0},{1},{2},{3},{4},{5},{6}'.format(
                int(obj_id),  # 0  # Appeared sometimes as float; intercepted here
                int(frame),  # 1
                x,  # 2
                y,  # 3
                w,  # 4
                h,  # 5
                deg  # 6
            )
            if illumination:
                curr_string = '{},{}\n'.format(curr_string, xy[2])
            else:
                curr_string = '{}\n'.format(curr_string)
            string_holder += curr_string  # append row
        with open(path, 'a', newline='') as file:  # append rows to .csv file
            file.write(string_holder)
    return None, None


def set_different_colour_filter(colour_filter_new):
    """
    sets a new cv2 colour filter

    :param colour_filter_new: name of colour filter
    :return: cv2 colour filter
    """
    logger = logging.getLogger('ysmr').getChild(__name__)
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
    preallocate empty array and assign slice of original array, filled with fill_value
    # See origin:
    # https://stackoverflow.com/questions/30399534/shift-elements-in-a-numpy-array
    # Accessed last 2019-04-24 13:37:00,101

    :param arr: array
    :type arr: np.array
    :param shift: shift amount, positive or negative
    :type shift: int
    :param fill_value: Value to fill up with, default np.nan
    :return: shifted array
    """
    result_array = np.empty_like(arr)
    if shift > 0:
        result_array[:shift] = fill_value
        result_array[shift:] = arr[:-shift]
    elif shift < 0:
        result_array[shift:] = fill_value
        result_array[:shift] = arr[-shift:]
    else:
        result_array[:] = arr
    return result_array


def sort_list(file_path=None, sort=None, df=None, save_file=False):
    """
    sorts pandas data frame, optionally saves it and loads it from csv

    :param file_path: file path to save .csv to
    :param sort: list of columns to sort by, defaults to ['TRACK_ID', 'POSITION_T']
    :type sort: list
    :param df: optional pandas data frame to be sorted
    :param save_file: whether to save the sorted file
    :type save_file: bool
    :return: sorted pandas data frame
    """
    logger = logging.getLogger('ysmr').getChild(__name__)
    if sort is None:
        sort = ['TRACK_ID', 'POSITION_T']
    elif isinstance(sort, (str, bytes)):
        sort = [sort]
    if file_path is not None and df is None:
        df = get_data(file_path, check_sorted=False)  # get data frame from .csv
    try:
        df.sort_values(by=sort, inplace=True, na_position='first')  # Sort data frame
        df.reset_index(drop=True, inplace=True)  # reset index of df
        logger.debug('Sorted data frame by {}.'.format(sort[0]))
    except Exception as ex:
        template = 'An exception of type {0} occurred while sorting file {2}. Arguments:\n{1!r}'
        logger.exception(template.format(type(ex).__name__, ex.args, file_path))
        return None
    if save_file and file_path is not None:
        save_df_to_csv(df=df, save_path=file_path, rename_old_file=False)
        # rename_old_file False as there will always be an old file
    elif save_file and file_path is None:
        logger.critical('Cannot save file if no file path is provided.')
    return df


def shutdown(seconds=60):
    """
    attempts to shut down the computer

    :param seconds: seconds before shutdown on windows
    :type seconds: int
    :return: None
    """
    logger = logging.getLogger('ysmr').getChild(__name__)
    if os.name is 'nt':  # windows
        try:
            response = subprocess.run('shutdown -f -s -t {}'.format(seconds), stderr=subprocess.PIPE)
            response.check_returncode()
            logger.warning('Calling \'shutdown -f -s -t {0}\' on system, '
                           'shutting down in {0} s'.format(seconds))
            logger.info('Type \'shutdown -a\' in command console to abort shutdown.')
        except (OSError, FileNotFoundError, subprocess.CalledProcessError) as os_shutdown_error:
            logger.exception('Error during shutdown: {}'.format(os_shutdown_error))
        finally:
            pass
    else:
        try:
            response = subprocess.run('systemctl poweroff', stderr=subprocess.PIPE)
            response.check_returncode()
            logger.warning('Calling \'systemctl poweroff\' on system.')
        except (OSError, FileNotFoundError, subprocess.CalledProcessError):
            try:
                response = subprocess.run('sudo shutdown -h +1', stderr=subprocess.PIPE)
                response.check_returncode()
                logger.warning('Calling \'sudo shutdown -h +1\' on system.')
            except (OSError, FileNotFoundError, subprocess.CalledProcessError) as os_shutdown_error:
                logger.exception('Error during shutdown: {}'.format(os_shutdown_error))
        finally:
            pass


if __name__ == '__main__':
    get_loggers(log_to_file=False, short_stream_output=True)
    create_configs()
