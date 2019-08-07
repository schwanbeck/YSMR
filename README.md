# YSMR
Bacterial video tracking and analysis


## Installation

### Dependencies
Python 3.5 or higher
#### Packages:
+ Matplotlib
+ Numpy
+ openCV v3 or v4; v2 untested (opencv-python or opencv-contrib-python)
+ Pandas
+ Scipy
+ Seaborn

Optional:
+ XlsxWriter

#### Files in this package:
+ \_\_init\_\_.py
    + empty
+ \_\_version\_\_.py
    + version file
+ helper_file.py
    + contains various functions used throughout YSMR
+ LICENSE
    + The license under which YSMR is published
+ plot_functions.py
    + plotting functions which are used to create graphs
+ REAMDE.md
    + this file
+ track_eval.py
    + tracking, selection, and evaluation functions of YSMR
+ tracker.py
    + the tracker used by YSMR
+ YSMR.py
    + The file which starts YSMR

## Usage

## Explanation of tracking.ini file
Explanation of each setting in the tracking.ini file and the initial default setting. The tracking.ini file will be newly generated when missing or if it cannot be loaded correctly.

##### BASIC RECORDING SETTINGS
+ video extension : .wmv
	+ When using display settings: 'select files' this specifies what file ending should be looked for in addition to the default formats .avi, .mp4, .mov.
	File selection can also be more specific, i.e. 'nice_video.avi', which would then only select files ending in nice_video.avi.
	+ Accepts text string.
+ pixel per micrometre : 1.41888781
	+ The ratio between pixels and micrometres. 
	+ Accepts a floating point number.
+ frames per second : 30.0
	+ The frames per second (fps) of the video file. Fallback value for when no fps is provided, for example when analysing .csv files directly.
	+ Automatic fps detection can be disabled under 'advanced video settings': 'force tracking.ini fps settings'.
	+ Accepts a floating point number.
+ frame height : 922
	+ The height of a frame in pixel. Fallback value for when no height dimension is provided, for example when analysing .csv files directly.
	+ Accepts an integer value.
+ frame width : 1228
	+ The width of a frame in pixel. Fallback value for when no width dimension is provided, for example when analysing .csv files directly.
	+ Accepts an integer value.
+ white bacteria on dark background : True
	+ If bacteria are displayed as white signals against a darker background, use 'True'. If the bacteria are darker than the average background, use 'False'.
	+ Accepts 'True' or 'False'.
+ rod shaped bacteria : True
	+ If bacteria are rod shaped, use 'True'. If the bacteria are coccoidal, set it to 'False'. This is a convenience boolean switch between two presets for length/width ratios. 
	+ Further width and height adjustments are available at 'advanced track data analyses settings'.
	+ Accepts 'True' or 'False'.
+ threshold offset for detection : 5
	+ Changes the grey value threshold which differentiates between bacteria and background. 
	Decrease value if bacteria are small and/or are not detected. 
	Increase value if background signals are detected as bacteria. 
	+ Accepts an integer value.

##### BASIC TRACK DATA ANALYSIS SETTINGS
+ minimal length in seconds : 20
	+ Minimum time threshold for track analysis. All tracks shorter than this will be discarded. If set to 0, all tracks will be accepted. 
	+ Accepts a floating point number.
+ limit track length to x seconds : 20
	+ Setting the value to 0 will disable the truncation.
	+ Maximum time for track analysis. All tracks longer than this will be truncated to the set time. If advanced track data analysis settings: 'track length exactly' is set to False (default) and the truncated track is not as long as the specified time due to holes in measurement, the nearest time point below the limit will be used instead. If 'track length exactly' is set to True, tracks that are shorter than the time specified after truncation will be discarded.  
	+ Accepts a floating point number.
+ minimal angle in degrees for turning point : 30.0
	+ Minimal change in angle (0° to 360°) between two positions in order to be declared a turning point.
	+ The difference in frames between the two comparison points are defined under 'advanced track data analysis settings': 'compare angle between n frames'.
	+ Accepts a floating point number.
+ extreme area outliers lower end in px*px : 2
	+ Lower limit for area in pixel^2 which is not considered as bacteria. Tracks with an average area below this value will be discarded entirely in an initial step.
	+ Accepts an integer value.
+ extreme area outliers upper end in px*px : 50
	+ Upper limit for area in pixel^2 which is not considered as bacteria. Tracks with an average area above this value will be discarded entirely in an initial step.
	+ Accepts an integer value.

##### DISPLAY SETTINGS
+ user input : True
	+ When set to True, after file selection and before starting with the analysis a command line prompt will wait for confirmation (y/n) to proceed, so that the user has time to review the selected files. If set to False, the process will proceed immediately.
	+ Accepts 'True' or 'False'.
+ select files : True
	+ When set to True, the user will be prompted to select a folder in which all video files with the specified file ending will be selected. 
	When set to False, the path specified under test settings: 'path to test video' will be used.
	+ Accepts 'True' or 'False'.
+ display video analysis : False
	+ When set to True, the video analysis process will be displayed in a window.
	+ Accepts 'True' or 'False'.
+ save video : False
	+ When set to True, the video analysis process will be saved in the same folder as the original video file.
	+ Accepts 'True' or 'False'.

##### RESULTS SETTINGS
+ rename previous result .csv : False
	+ When set to True, previously generated tracking result .csv files with the same name as newly generated files will be renamed (i.e. nice_video.avi_list.csv to nice_video.avi_list191224133700.csv). When set to False, previous tracking result .csv files with the same name will be overwritten.
	+ Accepts 'True' or 'False'.
+ delete .csv file after analysis : True
	+ When set to True, tracking result .csv files will be deleted after completion of analysis.
	+ Accepts 'True' or 'False'.
+ store processed .csv file : False
	+ When set to True, processed tracking result .csv files containing selected tracks will be saved in result folder after analysis.
+ store generated statistical .csv file : False
	+ When set to True, generated .csv files containing general track statistics will be saved in result folder after analysis.
	+ Accepts 'True' or 'False'.
+ save large plots : True
	+ When set to True a plot with an overview of the position of the tracked bacteria throughout the video will be generated.
	+ Accepts 'True' or 'False'.
+ save rose plot : True
	+ When set to True a graph in which the starting position of all tracks is set to (0,0) will be generated.
+ save time violin plot : True
	+ When set to True a violin plot of the track times will be generated.
	+ Accepts 'True' or 'False'.
+ save acr violin plot : False
	+ When set to True a violin plot of the tracks arc-chord ratio will be generated.
	+ Accepts 'True' or 'False'.
+ save length violin plot : True
	+ When set to True a violin plot of the tracks length will be generated.
	+ Accepts 'True' or 'False'.
+ save turning point violin plot : True
	+ When set to True a violin plot of the tracks turning points will be generated.
	+ Accepts 'True' or 'False'.
+ save speed violin plot : True
	+ When set to True a violin plot of the tracks average speed will be generated.
	+ Accepts 'True' or 'False'.
+ save angle distribution plot / bins: 36
	+ Setting the value to 0 will disable the angle distribution plot.
	+ ### Angle distribution plot description here
	+ Accepts 'True' or 'False'.
+ collate results csv to xlsx: True
	+ ### distribution here
	+ Accepts 'True' or 'False'.

##### LOGGING SETTINGS
+ log to file : True
	+ When set to True, the logging output will be saved to file.
	+ Accepts 'True' or 'False'.
+ log file path : ./logfile.log
	+ The log-file save path.
	+ Accepts text string.
+ shorten displayed logging output : False
	+ When set to True, the console logging output will be shortened to 'Time	level-name	PID	message'
	+ Accepts 'True' or 'False'.
+ shorten logfile logging output : False
	+ When set to True, the logfile logging output will be shortened to 'Time	level-name	PID	message'
	+ Accepts 'True' or 'False'.
+ set logging level (debug/info/warning/critical) : debug
	+ Set the logging level. Fallback value is 'debug'.
	+ Accepts 'debug', 'info', 'warning', 'critical'.
+ verbose : False
	+ When set to True the log-level is set to debug and logs additional debug messages.
	+ Accepts 'True' or 'False'.

##### ADVANCED VIDEO SETTINGS
+ use default extensions (.avi, .mp4, .mov) : True
	+ When set to true and display settings: 'select files' is set to True, files ending in .avi, .mp4, and .mov will also be selected in addition to the file ending specified under basic recording settings: 'video extension'.
	+ Accepts 'True' or 'False'.
+ include luminosity in tracking calculation : False
	+ When set to True, in addition to the x- and y-position of the bacteria, luminosity will be used as an additional dimension during tracking.
	+ Accepts 'True' or 'False'.
+ color filter : COLOR_BGR2GRAY
	+ The colour filter conversion used by openCV. Should convert to a grey-scale image.
	+ Accepts text string of openCV conversion flag or corresponding integer value.
+ maximal video file age (infinite or seconds) : infinite
	+ When a value in seconds is given, files older than the specified time will not be used for analysis.
	+ Depends on 'display settings': 'select files'
	+ Accepts an integer value or 'infinite'.
+ minimal video file age in seconds : 0
	+ When a value in seconds is given, files younger than the specified time will not be used for analysis.
	+ When set to any negative value, files with a calculated negative age will also be accepted.
	+ Depends on 'display settings': 'select files'
	+ Accepts an integer value.
+ minimal frame count : 600
	+ Minimal frame count of video files. Files with a shorter frame count will be skipped.
	+ Accepts an integer value.
+ stop evaluation on error : True
	+ When set to True, if a presumed file reading error is encountered during video analysis, evaluation of the generated tracking result .csv file will be skipped.
	+ Accepts 'True' or 'False'.
+ list save length interval : 10000
	+ Minimal length of tracking result object list before it will be stored on disc.
	+ Accepts an integer value.
+ force tracking.ini fps settings : False
	+ When set to True, the frames per second specified in tracking.ini under basic recording settings: 'frames per second' will be used instead of the fps information provided by the video file.
	+ Accepts 'True' or 'False'.

##### ADVANCED TRACK DATA ANALYSIS SETTINGS
+ maximal consecutive holes : 5
	+ Maximal consecutively missing values in track. If a hole greater than specified is encountered, the track will be split at the hole and the parts re-analysed.
	+ Accepts an integer value.
+ maximal empty frames in % : 5.0
	+ Maximal total percentage of missing values.
	+ Accepts a floating point number.
+ percent quantiles excluded area : 10.0
	+ Setting the value to 0 will disable exclusion.
	+ When a value other than 0 is given, the upper and lower quantile (i.e. here 10 % and 90 %) of average track area will be excluded.
	+ Accepts a positive floating point number.
+ try to omit motility outliers : True
	+ When set to True, tracks will be split on distance values above the outer fence (3 * IQR plus the 75% quartile) of each track, as long as the total number of such events is below the threshold specified under 'stop excluding motility outliers if total count above percent'.
	+ Accepts 'True' or 'False'.
+ stop excluding motility outliers if total count above percent : 5.0
	+ The limit for 'try to omit motility outliers'. If the percentage of calculated motility outliers compared to all data points surpasses the given percentage, 'try to omit motility outliers' will be retroactively disabled.
	+ Accepts a floating point number.
+ exclude measurement when above x times average area : 2.0
	+ Setting the value to 0 will disable exclusion.
	+ Data points with area measurements above x times the average area of the track will be discarded.
	+ Accepts a floating point number.
+ rod average width/height ratio min. : 0.125
	+ Lower limit average width to height ratio when basic recording settings: 'rod shaped bacteria' is set to True. Example: the default value is equal to a ratio of 1:8.
	+ Accepts a floating point number.
+ rod average width/height ratio max. : 0.67
	+ Upper limit average width to height ratio when basic recording settings: 'rod shaped bacteria' is set to True. Example: the default value is equal to a ratio of ~1:1.5.
	+ Accepts a floating point number.
+ coccoid average width/height ratio min. : 0.8
	+ Lower limit average width to height ratio when basic recording settings: 'rod shaped bacteria' is set to False. Example: the default value is equal to a ratio of 1:1.25.
	+ Accepts a floating point number.
+ coccoid average width/height ratio max. : 1.0
	+ Upper limit average width to height ratio when basic recording settings: 'rod shaped bacteria' is set to False. Example: the default value is equal to a ratio of 1:1.
	+ Accepts a floating point number.
+ percent of screen edges to exclude : 5.0
	+ Setting the value to 0 will disable exclusion.
	+ Exclude tracks if their average x- or y-position is within specified percent of the screen edges.
	+ Accepts a floating point number.
+ maximal recursion depth : 960
	+ Setting the value to 0 will disable recursion.
	+ Tracks will be split on erroneous data (advanced track data analysis settings: 'maximal consecutive holes' or 'try to omit motility outliers') and parts recursively analysed until the specified depth is reached.
	+ Accepts an integer value.
+ limit track length exactly : False
	+ See basic track data analysis settings: 'limit track length to x seconds'.
	+ When set to False and the truncated track is not as long as the specified time in 'limit track length to x seconds' due to holes in measurement, the nearest time point below the limit will be used instead. When set to True, tracks that are shorter than the time specified in 'limit track length to x seconds' after truncation will be discarded. 
	+ Accepts 'True' or 'False'.
+ compare angle between n frames : 10
	+ See basic track data analysis settings: 'minimal angle in degrees for turning point'
	+ Difference in frames between which position of bacteria and corresponding angle will be measured.
	+ Time difference (measured in frames) between bacterial position in between which angle is measured.
	+ Accepts an integer value.

##### HOUSEKEEPING
+ previous directory : ./
	+ When selecting files, the last specified folder will be used as starting point.
	+ Fallback value is './'.
+ shut down after analysis : False
	+ Attempt to shut down the OS after analysis has finished.
	+ Accepts 'True' or 'False'.

##### TEST SETTINGS
+ debugging : False
	+ 
	+ ### When set to True, this will directly start with the analysis of the specified test video file. When display settings: 'display video analysis' is set to True, it will additionally display the threshold of the analysed video file.
	+ Accepts 'True' or 'False'.
+ path to test video : Q:/test_video.avi
	+ Accepts text string.
+ path to test .csv : Q:/test_list.csv
	+ Accepts text string.

## Citation

```
@Article{
  doi = {}
  url = {https},
  year  = {2019},
  month = {},
  publisher = {},
  author = {},
  editor = {},
  title = {},
  journal = {}
}
```

## Acknowledgements
The original tracker.py was taken with permission from Adrian Rosebrock (adrian@pyimagesearch.com) from https://www.pyimagesearch.com/2018/07/23/simple-object-tracking-with-opencv/
