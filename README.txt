# YSMR
Bacterial video tracking &amp; analysis


## Installation

## Usage

## Explanation of tracking.ini file
Explanation of each setting in the tracking.ini file and the initial default setting. The tracking.ini file will be generated when missing or if it cannot be loaded correctly.

##### BASIC RECORDING SETTINGS
+ video extension : .mp4
	+ When using 'select files' this specifies what file ending should be looked for. This format will be used in addition to the formats .avi, .mp4, .mov. This behaviour can be changed under advanced video settings: 'use default extensions (.avi, .mp4, .mov)'. This can also be more specific, ie. 'nice_video.avi', which would then only select files ending in nice_video.avi.
	+ Accepts text string.
+ pixel per micrometre : 1.41888781
	+ The ratio between pixels and micrometres.
	+ Accepts a floating point number.
+ frames per second : 30.0
	+ The frames per second (fps) at which the movie was captured. If a video file is analysed, the encoded fps value will be used preferentially, if available. This can be disabled under advanced video settings: 'force tracking.ini fps settings'.
	+ Accepts a floating point number.
+ frame height : 922
	+ The height of a frame in pixel. If a video file is analysed, the encoded height value will be used preferentially, if available.
	+ Accepts an integer value.
+ frame width : 1228
	+ The width of a frame in pixel. If a video file is analysed, the encoded width value will be used preferentially, if available.
	+ Accepts an integer value.
+ white bacteria on dark background : True
	+ If bacteria show up as white against a darker background, this setting should be set to 'True'. If the bacteria are darker than the average background, it should be 'False'.
	+ Accepts 'True' or 'False'.
+ rod shaped bacteria : True
	+ If rod shaped bacteria are filmed, set this to 'True'. If the bacteria are coccoidal, set it to 'False'. This is a convenience boolean switch between two presets for length/width ratios.
	+ See the under advanced track data analysis settings:
		+ 'rod average width/height ratio min.'
		+ 'rod average width/height ratio max.'^
		+ 'coccoid average width/height ratio min.'
		+ 'coccoid average width/height ratio max.'
	+ Accepts 'True' or 'False'.
+ threshold offset for detection : 5
	+ Changes the grey value threshold which differentiates between bacteria and background. Decrease if bacteria are often too small or not detected. Increase if background is mistaken for bacteria during detection.
	+ Accepts an integer value.

##### BASIC TRACK DATA ANALYSIS SETTINGS
+ minimal length in seconds : 5
	+ Minimum time threshold for track analysis. All tracks shorter than this will be discarded. If set to 0, all tracks will be accepted.
	+ Accepts an integer value.
+ limit track length to x seconds : 20
	+ Setting the value to 0 will disable the truncation.
	+ Maximum time for track analysis. All tracks longer than this will be truncated to the set time. If advanced track data analysis settings: 'track length exactly' is set to False (default) and the truncated track is not as long as the specified time due to holes in measurement, the nearest time point below the limit will be used instead. If 'track length exactly' is set to True, tracks that are shorter than the time specified after truncation will be discarded.
	+ Accepts an integer value.
+ minimal angle in degrees for turning point : 30.0
	+ Minimal change in angle between two positions in order to be declared a turning point. All relative changes in angle below will be viewed as motile.
	+ The difference in frames between the two comparison points are defined under advanced track data analysis settings: 'compare angle between n frames'.
	+ Accepts a floating point number.
+ extreme size outliers lower end in px : 2
	+ Lower limit for area (pixel^2) which is not considered as bacteria. Tracks with an average area below this value will be discarded entirely in an initial step.
	+ Accepts an integer value.
+ extreme size outliers upper end in px : 50
	+ Upper limit for area (pixel^2) which is not considered as bacteria. Tracks with an average area above this value will be discarded entirely in an initial step.
	+ Accepts an integer value.

##### DISPLAY SETTINGS
+ user input : True
	+ When set to True, after file selection and before starting with the analysis a command line prompt will wait for confirmation (y/n) to proceed, so that the user has time to review the selected files. If set to False, the process will proceed immediately.
	+ Accepts 'True' or 'False'.
+ select files : True
	+ When set to True, the user will be prompted to select a folder in which all video files with the correct file ending will be selected. When set to False, the path specified under test settings: 'path to test video' will be used.
	+ Accepts 'True' or 'False'.
+ display video analysis : False
	+ When set to True, the video analysis process will be displayed in a window.
	+ Accepts 'True' or 'False'.
+ save video : False
	+ When set to True, the video analysis process will be saved in the same folder as the original video file.
	+ Accepts 'True' or 'False'.

##### RESULTS SETTINGS
+ rename previous result .csv : False
	+ When set to True, previously generated tracking result .csv files with the same name as newly generated files will be renamed. When set to False, previous tracking result .csv files with the same name will be deleted.
	+ Accepts 'True' or 'False'.
+ delete .csv file after analysis : True
	+ When set to True, tracking result .csv files will be deleted after completion of analysis.
	+ Accepts 'True' or 'False'.
+ store processed .csv file : False
	+ When set to True, processed tracking result .csv files containing selected tracks will be saved on disc after analysis.
+ store generated statistical .csv file : False
	+ When set to True, generated .csv files containing general track statistics will be saved on disc after analysis.
	+ Accepts 'True' or 'False'.
+ save large plots : True
	+ When set to True a plot with an overview of the position of the tracked bacteria throughout the video will be generated.
	+ Accepts 'True' or 'False'.
+ save rose plot : True
	+ When set to True a graph in which each tracks starting position is set to (0,0) will be generated.
+ save time violin plot : True
	+ When set to True a violin plot of the track times will be generated.
	+ Accepts 'True' or 'False'.
+ save acr violin plot : False
	+ When set to True a violin plot of the tracks arc-chord ratio will be generated.
	+ Accepts 'True' or 'False'.
+ save distance violin plot : True
	+ When set to True a violin plot of the tracks distances will be generated.
	+ Accepts 'True' or 'False'.
+ save turning point violin plot : True
	+ When set to True a violin plot of the tracks turning points will be generated.
	+ Accepts 'True' or 'False'.
+ save speed violin plot : True
	+ When set to True a violin plot of the tracks speed will be generated.
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
	+ When set to True sets the log-level to debug and logs additional debug messages.
	+ Accepts 'True' or 'False'.

##### ADVANCED VIDEO SETTINGS
+ use default extensions (.avi, .mp4, .mov) : True
	+ When set to true and display settings: 'select files' is set to True, files ending in .avi, .mp4, and .mov will also be selected in addition to the file ending specified under basic recording settings: 'video extension'.
	+ Accepts 'True' or 'False'.
+ include luminosity in tracking calculation : False
	+ When set to True, in addition to the x- and y-position of the bacteria, luminosity will be used as an additional dimension during tracking.
	+ Accepts 'True' or 'False'.
+ color filter : COLOR_BGR2GRAY
	+ The colour filter conversion used by openCV. Should end up with a grey-scale image.
	+ Accepts text string of openCV conversion flag or corresponding integer value.
+ maximal video file age (infinite or seconds) : infinite
	+ When not set to infinite, video files found if display settings: 'select files' is set to True older than the time specified will not be used.
	+ Accepts an integer value or 'infinite'.
+ minimal video file age in seconds : 0
	+ Video files found if display settings: 'select files' is set to True younger than the time specified will not be used.
	+ Accepts an integer value.
+ minimal frame count : 600
	+ Minimal frame count of video file. Files with a shorter frame count will be skipped.
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
	+ Maximal consecutively missing values in track. If a hole greater than specified is encountered, the track will be split on the hole and the parts re-analysed.
	+ Accepts an integer value.
+ maximal empty frames in % : 5.0
	+ Maximal total percentage of missing values.
	+ Accepts a floating point number.
+ percent quantiles excluded area : 10.0
	+ Setting the value to 0 will disable it.
	+ When a value other than 0 is given, the upper and lower quantile (i.e. here 10 % and 90 %) of average track area will be excluded.
	+ Accepts a positive floating point number.
+ try to omit motility outliers : True
	+ When set to True, tracks will be split on distance values above the outer fence (3 * IQR plus the 75% quartile) of each track, as long as the total number of such events is below the threshold specified under 'stop excluding motility outliers if total count above percent'.
	+ Accepts 'True' or 'False'.
+ stop excluding motility outliers if total count above percent : 5.0
	+ The limit for 'try to omit motility outliers'. If the percentage of calculated motility outliers compared to all data points surpasses the given percentage, 'try to omit motility outliers' will be retroactively disabled.
	+ Accepts a floating point number.
+ exclude measurement when above x times average area : 2.0
	+ Setting the value to 0 will disable it.
	+ Data points with area measurements above x times the average area of the track will be discarded.
	+ Accepts a floating point number.
+ rod average width/height ratio min. : 0.125
	+ Lower end average width to height ratio when basic recording settings: 'rod shaped bacteria' is set to True. Example: the default value is equal to a ratio of 1:8.
	+ Accepts a floating point number.
+ rod average width/height ratio max. : 0.67
	+ Upper end average width to height ratio when basic recording settings: 'rod shaped bacteria' is set to True. Example: the default value is equal to a ratio of ~1:1.5.
	+ Accepts a floating point number.
+ coccoid average width/height ratio min. : 0.8
	+ Lower end average width to height ratio when basic recording settings: 'rod shaped bacteria' is set to False. Example: the default value is equal to a ratio of 1:1.25.
	+ Accepts a floating point number.
+ coccoid average width/height ratio max. : 1.0
	+ Upper end average width to height ratio when basic recording settings: 'rod shaped bacteria' is set to False. Example: the default value is equal to a ratio of 1:1.
	+ Accepts a floating point number.
+ percent of screen edges to exclude : 5.0
	+ Setting the value to 0 will disable it.
	+ Exclude tracks if their average x- or y-position is within x percent of the screen edges.
	+ Accepts a floating point number.
+ maximal recursion depth : 960
	+ Setting the value to 0 will disable it.
	+ Tracks will be split on errors (advanced track data analysis settings: 'maximal consecutive holes' or 'try to omit motility outliers') and parts recursively analysed until the specified depth is reached.
	+ Accepts an integer value.
+ limit track length exactly : False
	+ See basic track data analysis settings: 'limit track length to x seconds'.
	+ When set to False and the truncated track is not as long as the specified time in 'limit track length to x seconds' due to holes in measurement, the nearest time point below the limit will be used instead. When set to True, tracks that are shorter than the time specified in 'limit track length to x seconds' after truncation will be discarded.
	+ Accepts 'True' or 'False'.
+ compare angle between n frames : 10
	+ See basic track data analysis settings: 'minimal angle in degrees for turning point'
	+ Difference in frames between which position of bacteria and corresponding angle will be measured.
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
	+ When set to True, this will directly start with the analysis of the specified test video file. When display settings: 'display video analysis' is set to True, it will additionally display the threshold of the analysed video file.
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
