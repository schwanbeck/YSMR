import logging

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt


def angle_distribution_plot(df, bins_number, plot_title_name, save_path):
    logger = logging.getLogger('ei').getChild(__name__)
    angle_radians = df['angle_diff']
    # Create array with average motility percentage per track
    average_minimum_groups = df.groupby('TRACK_ID')['minimum']
    min_average = np.repeat(average_minimum_groups.mean().to_numpy(), average_minimum_groups.count().to_numpy())
    # Kick out all tracks with less than 70 % motility  @todo: make conditional
    angle_radians_minimum = np.where(
        min_average > 0.7,
        df['minimum'],
        0
    ).astype(dtype=bool)

    # If there are any data points (0 == False):
    if not angle_radians_minimum.sum():
        logger.warning('Cannot create angle distribution plot as there are no motile tracks.')
        return

    all_angles_radians = angle_radians[np.array(angle_radians_minimum)]  # , dtype=bool
    # print(stats.describe(all_angles_radians, nan_policy='omit'))
    bins = np.linspace(-np.pi, np.pi, bins_number + 1)
    hist_array, _ = np.histogram(all_angles_radians, bins)
    # plt.clf()
    plt.figure(figsize=(11.6929133858, 8.2677165354))  # , gridspec_kw={'width_ratios': [1, 1, 1, 1]}
    ax = plt.subplot(1, 1, 1, projection='polar')
    ax.set_theta_zero_location("N")  # theta=0 at the top
    ax.set_theta_direction(-1)  # theta increasing clockwise
    width = 2 * np.pi / bins_number
    bars = ax.bar(
        bins[:bins_number],
        hist_array,
        # height=,
        width=width,
        bottom=0.0,
        # align='center',  # 'edge',  # x axis: 'edge': Align the left edges of the bars with the x positions.
        # color='blue',
        edgecolor='k',
        # linewidth=,
        # tick_label=,
        # eclolr=,  # error bar color
        # capsize=,  # error bar cap length
        # error_kw=,  # dict of kwargs for error bar
        # log=False,  # logarithmic y-axis
    )
    for bar in bars:
        bar.set_alpha(0.5)
    plt.title('{} Data points: {}'.format(plot_title_name, angle_radians_minimum.sum()))
    plt.savefig(save_path, dpi=300)
    logger.debug('Saving figure {}'.format(save_path))
    plt.close()


def save_large_plot(df, plot_title_name, track_count, save_path):
    logger = logging.getLogger('ei').getChild(__name__)
    # DIN A4, as used in the civilised world  # @todo: let user select other, less sophisticated, formats
    plt.figure(figsize=(11.6929133858, 8.2677165354))  # , gridspec_kw={'width_ratios': [1, 1, 1, 1]}
    plt.grid(True)
    plt.axis('equal')
    # display initial position as black dots
    grouped_df = df.groupby('TRACK_ID')['POSITION_X', 'POSITION_Y'].transform('first')
    plt.scatter(
        grouped_df.POSITION_X,
        grouped_df.POSITION_Y,
        marker='o',
        color='black',
        s=1,
        lw=0,
    )
    # Group df by TRACK_ID, sort by distance_colour, leave x/y coordinates and colour, plot those
    grouped_df = df.loc[:, ['TRACK_ID', 'distance_colour', 'POSITION_X', 'POSITION_Y', ]].sort_values(
        ['distance_colour'], ascending=False
    ).groupby('TRACK_ID', sort=False)['POSITION_X', 'POSITION_Y', 'distance_colour']

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
    plt.title('{} Track count: {}'.format(plot_title_name, track_count))
    # save_path = '{}{}_Bac_Run_Overview.png'.format(results_directory, file_name)
    plt.savefig(save_path, dpi=300)
    logger.debug('Saving figure {}'.format(save_path))
    plt.close()


"""
settings['save angle distribution plot / bins']
settings['store processed .csv file']
settings['store generated statistical .csv file']
settings['save large plots']


'save rose plot'
'save time violin plot'
'save acr violin plot'
'save length violin plot'
'save turning point violin plot'
'save speed violin plot'
'collate results csv to xlsx'
-> collate results .csv to xlsx + delete .csv
"""
