#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Copyright 2019, 2020 Julian Schwanbeck (julian.schwanbeck@med.uni-goettingen.de)
https://github.com/schwanbeck/YSMR
##Explanation
This file contains functions which create plots for YSMR.
This file is part of YSMR. YSMR is free software: you can distribute it and/or modify
it under the terms of the GNU General Public License as published by the Free Software Foundation,
either version 3 of the License, or (at your option) any later version. YSMR is distributed in
the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
details. You should have received a copy of the GNU General Public License along with YSMR. If
not, see <http://www.gnu.org/licenses/>.
"""

import logging

import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def angle_distribution_plot(df, bins_number, plot_title_name, save_path, dpi=300):
    """
    bins all angles and plots them on a circle
    :param df: pandas data frame with angles
    :param bins_number: count of bins
    :type bins_number: int
    :param plot_title_name: name of plot
    :type plot_title_name: str
    :param save_path: path to save image
    :param dpi: dpi of saved image
    :type dpi: int
    :return: None
    """
    logger = logging.getLogger('ysmr').getChild(__name__)
    angle_radians = df['angle_diff']
    # Create array with average motility percentage per track
    average_minimum_groups = df.groupby('TRACK_ID')['moving']
    min_average = np.repeat(average_minimum_groups.mean().to_numpy(), average_minimum_groups.count().to_numpy())
    # Kick out all tracks with less than 70 % motility  @todo: make conditional
    angle_radians_minimum = np.where(
        min_average > 0.7,
        df['moving'],
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
    plt.savefig(save_path, dpi=dpi)
    logger.debug('Saving figure {}'.format(save_path))
    plt.close()


def colour_bar(ax, dist_min, dist_max):
    """
    Adds a colour bar element to axis

    :param ax: matplotlib axis
    :param dist_min: minimal value
    :param dist_max: maximal value
    :return: matplotlib axis
    """
    colorbar_map = plt.cm.viridis_r
    norm = mpl.colors.Normalize(vmin=dist_min, vmax=dist_max)
    cb = mpl.colorbar.ColorbarBase(ax, cmap=colorbar_map, norm=norm, )
    cb.set_label('\u00B5m')
    return ax


def large_xy_plot(df, plot_title_name, save_path, px_to_micrometre=1, dist_min=0, dist_max=None, dpi=300):
    """
    save x/y-coordinates through time off all tracks on one plot

    :param df: pandas data frame with x, y, time coordinates
    :param plot_title_name: name of plot
    :type plot_title_name: str
    :param save_path: path to save image
    :param px_to_micrometre: ratio between pixel and micrometre
    :type px_to_micrometre: float
    :param dist_min: minimal distance for distance bar
    :type dist_min: float
    :param dist_max: maximal distance for distance bar
    :param dpi: dpi of saved image
    :type dpi: int
    :return: None
    """
    logger = logging.getLogger('ysmr').getChild(__name__)

    f = plt.figure()
    f.set_size_inches(11.6929133858, 8.2677165354)

    outer_space = 0.05
    # inner_space = 0.03
    head_space = 0.05
    width_space = 0.05

    # plt.rcParams.update({'font.size': 8})
    plt.rcParams['axes.axisbelow'] = True

    gs = gridspec.GridSpec(1, 100, figure=f)
    gs.update(left=outer_space, right=1 - outer_space, hspace=head_space, wspace=width_space)

    if not dist_max:
        try:
            dist_max = df['travelled_dist'].max()
        except KeyError:
            dist_max = df['distance_colour'].max()

    large_plot = plt.subplot(gs[0, :-2])  # xy-centered plots
    dist_bar = plt.subplot(gs[0, -2:])  # distance color-map
    # display initial position as black dots
    grouped_df = df.groupby('TRACK_ID')[['POSITION_X', 'POSITION_Y']].transform('first')
    large_plot.scatter(
        grouped_df.POSITION_X / px_to_micrometre,
        grouped_df.POSITION_Y / px_to_micrometre,
        marker='o',
        color='black',
        s=1,
        lw=0,
    )
    # Group df by TRACK_ID, sort by distance_colour, leave x/y coordinates and colour, plot those
    grouped_df = df.loc[:, ['TRACK_ID', 'distance_colour', 'POSITION_X', 'POSITION_Y', ]].sort_values(
        ['distance_colour'], ascending=False
    ).groupby('TRACK_ID', sort=False)[['POSITION_X', 'POSITION_Y', 'distance_colour']]
    track_count = 0
    for name, group in grouped_df:
        track_count += 1
        large_plot.scatter(
            group.POSITION_X / px_to_micrometre,
            group.POSITION_Y / px_to_micrometre,
            marker='.',
            label=name,
            c=plt.cm.viridis_r(group.distance_colour),
            # vmin=distance_min,
            # vmax=distance_max,
            # cmap=plt.cm.viridis_r,
            s=1,
            lw=0,
        )
    colour_bar(ax=dist_bar, dist_min=dist_min, dist_max=dist_max)
    # plt.title('{} Track count: {}'.format(plot_title_name, track_count))
    large_plot.set_xlabel('\u00B5m')
    large_plot.set_ylabel('\u00B5m')
    large_plot.set_aspect('equal')
    large_plot.grid(True)
    large_plot.set_title('{}'.format(plot_title_name, ))
    plt.savefig(save_path, dpi=dpi)
    logger.debug('Saving figure {}'.format(save_path))
    plt.close()


def rose_graph(df, plot_title_name, save_path, dist_min=0, dist_max=None, dpi=300):
    """
    saves plot of all frames centered with initial x/y-coordinates set to 0,0

    :param df: pandas data frame
    :param plot_title_name: name of plot
    :type plot_title_name: str
    :param save_path: path to save image
    :param dist_min: minimal distance travelled
    :type dist_min: float
    :param dist_max: maximal distance travelled
    :type dist_max: float
    :param dpi: dpi of saved image
    :type dpi: int
    :return: None
    """
    logger = logging.getLogger('ysmr').getChild(__name__)
    if not dist_max:
        try:
            dist_max = df['travelled_dist'].max()
        except KeyError:
            dist_max = df['distance_colour'].max()
    # set up figure
    f = plt.figure()
    f.set_size_inches(11.6929133858, 8.2677165354)

    outer_space = 0.05
    # inner_space = 0.03
    head_space = 0.05
    width_space = 0.05

    # plt.rcParams.update({'font.size': 8})
    plt.rcParams['axes.axisbelow'] = True

    gs = gridspec.GridSpec(1, 100, figure=f)
    gs.update(left=outer_space, right=1 - outer_space, hspace=head_space, wspace=width_space)

    rose_plot = plt.subplot(gs[0, :-2])  # xy-centered plots
    dist_bar = plt.subplot(gs[0, -2:])  # distance color-map
    # get relevant columns, sort by distance (descending), then group sorted df by TRACK_ID
    grouped_df = df.loc[
                 :, ['TRACK_ID', 'distance_colour', 'x_norm', 'y_norm']
                 ].sort_values(['distance_colour'], ascending=False).groupby(
        'TRACK_ID', sort=False)[['x_norm', 'y_norm', 'distance_colour']]
    # @todo: Circles indicate the mean and 90th percentile net displacements
    for name, group in grouped_df:
        rose_plot.scatter(
            group.x_norm,
            group.y_norm,
            marker='.',
            label=name,
            c=plt.cm.viridis_r(group.distance_colour),
            # vmin=distance_min,
            # vmax=distance_max,
            # cmap=plt.cm.viridis_r,
            s=1,
            lw=0,
        )
    rose_plot.set_aspect('equal')
    rose_plot.grid(True)
    rose_plot.set_title('{}'.format(plot_title_name, ))

    # Distance colour indicator bar
    colour_bar(ax=dist_bar, dist_min=dist_min, dist_max=dist_max)
    plt.savefig(save_path, dpi=dpi)
    logger.debug('Saving figure {}'.format(save_path))
    plt.close()


def violin_plot(df, save_path, category, cut_off_category, cut_off_list, plot_title_name='\n\n',
                axis=None, dpi=300, verbose=False, y_min=None, y_max=None, ):
    """
    Create a violin plot

    :param df: pandas data frame
    :param save_path: output save path
    :param category: for y-axis
    :param cut_off_category: for x-axis
    :param cut_off_list: list of names for categories in x-axis
    :param plot_title_name: plot title
    :param axis: optional matplotlib axis on which to draw
    :param dpi: dpi of image
    :param verbose: verbose logging
    :param y_min: minimum for y-axis
    :param y_max: maximum for y-axis
    :type verbose: bool
    :return: None
    """
    logger = logging.getLogger('ysmr').getChild(__name__)
    # large = 12
    med = 8
    small = 6
    params = {
        'axes.titlesize': med,
        'legend.fontsize': med,
        # 'figure.figsize': (8.2677165354, 2.5),
        'axes.labelsize': small,
        # 'axes.titlesize': med,
        'xtick.labelsize': med,
        'ytick.labelsize': med,
        'figure.titlesize': med
    }
    plt.rcParams.update(params)
    plt.style.use('seaborn-whitegrid')
    save_fig = False
    if axis is None:  # in case it's not plotted/saved directly
        fig = plt.figure()
        fig.set_size_inches(11.6929133858 / 2, 8.2677165354 / 2)
        axis = fig.add_subplot(111)  # so we have an axis element
        save_fig = True
    # df_curr = df.loc[df[cut_off_category] != 'All']
    plt.rcParams['axes.axisbelow'] = True
    axis.grid(axis='y',
              which='major',
              # color='gray',
              alpha=0.80, )
    sns.violinplot(y=df[category],
                   x=df[cut_off_category],
                   # hue=df_stats[name_of_columns[-1]],
                   # dodge=False,
                   orient='v',
                   cut=0,
                   ax=axis,
                   scale='count',  # 'width' 'count' 'area'
                   width=0.95,
                   linewidth=1,
                   bw=.2,
                   # inner='stick',
                   )
    axis.set(ylim=(y_min, y_max))
    # Remove top and right border
    sns.despine(ax=axis, offset=0)
    # Empty title so there's 2 lines of space for the text boxes
    axis.set_title('{}\n\n'.format(plot_title_name))
    # set up data for text boxes
    text_boxes = []
    all_entries = sum(df[cut_off_category] == cut_off_list[0][2])
    if all_entries == 0:
        all_entries = df.shape[0]
    for idx_textbox in range(len(cut_off_list)):
        curr_category = cut_off_list[idx_textbox][2]
        curr_entries = sum(df[cut_off_category] == curr_category)
        df_subset = df.loc[df[cut_off_category] == curr_category, category]
        median = df_subset.median()
        average = df_subset.mean()
        if np.isnan(median):
            continue
        if all_entries > 0:
            curr_percentage = '{:.1%}'.format(curr_entries / all_entries)
        else:
            curr_percentage = 'error'
        text_boxes.append((curr_category, curr_percentage, median, average))

    # Create description (title) for each violin plot
    for idx_textbox, (curr_category, curr_percentage, qm_plot, average_plot) in enumerate(text_boxes):
        axis.text(
            idx_textbox / len(text_boxes) + 0.015, 1.005,
            '{}: {}\nMedian: {:.2f}\nAverage:  {:.2f}'.format(
                curr_category, curr_percentage,
                qm_plot,
                average_plot),
            # Set Textbox to relative position instead of absolute xy coordinates (0-1
            transform=axis.transAxes,
            size=small,
        )
    if save_fig:
        plt.savefig(save_path, dpi=dpi)
        if verbose:
            logger.debug('Saving figure {}'.format(save_path))
        plt.close()
    else:
        return axis
