# -*- coding: UTF-8 -*-
"""
This module contains functions for plotting the results of evaluation experiments.
"""

import os

import matplotlib.pyplot as plt


def plot_config(base_dir, sub_dirs=[], prefix="", img_type="pdf"):
    """
    Configure where plots should be stored and which file type should be used.
    @param base_dir: The base dir where all plots should be stored, e.g. ../plots
    @param sub_dirs: A list of dirs that will be concatenated to create the actual plot dir,
    e.g. sub_dirs=["houseA","scatter"] and base_dir="../plots", the images will be stored in "../plots/houseA/scatter".
    @param prefix: A prefix to append to each plot file name.
    @param img_type: The file type to use for storing the plot image, must be supported by pyplot.
    @return: A function that can be called to get the full path for a plot file.
    """
    plot_dir = base_dir
    for sub_dir in sub_dirs:
        plot_dir = os.path.join(plot_dir, sub_dir)

    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    def full_plot_path(name):
        prefixed_name = "%s%s.%s" %(prefix, str(name), img_type)
        return os.path.join(plot_dir, prefixed_name)

    return full_plot_path


def plot_quality_comparison(results, metric, plot_path):
    """
    Create a lineplot, with one line for every evaluated classifier, showing the measured metric for this classifier.
    @param results: A pandas dataframe with one column for every classifier, containing measurements for the metric.
    @param metric: The metric to plot.
    @param plot_path: A function that can take a local file name and give back the full path to that file,
    @return: None
    """
    plt.figure(figsize=(6, 6), dpi=300)
    plt.ylabel(metric, fontsize=18)
    plt.xlabel('recommendation cutoff', fontsize=18)
    results.plot(marker=".", colormap="prism")
    if not metric == "# of recommendations":
        plt.ylim(0.0, 1.0)
    plt.xlim(1, plt.xlim()[1])
    plt.legend(tuple(results.columns.values), loc=4)
    plt.savefig(plot_path(metric))
    plt.close()


def plot_train_size(results, metric, plot_path):
    """
    Create a lineplot, that shows how training size influences the given metric, with one line for every evaluated
    classifier.
    @param results: A pandas dataframe with one column for every classifier, containing measurements for the metric. The
    dataframe should have a multiIndex consisting of a tuple (size of training data, elapsed time).
    @param metric: The name of the metric that is to plot.
    @param plot_path:  A function that can take a local file name and give back the full path to that file.
    @return: None
    """
    #only want to have elapsed time as an index
    results.index = results.index.droplevel(0)

    plt.figure(figsize=(6, 6), dpi=300)
    plt.subplots_adjust(bottom=0.15)
    plt.ylabel(metric, fontsize=18)
    plt.xlabel('Elapsed training time (days)', fontsize=18)
    if not metric == "# of recommendations":
        plt.ylim(0.0, 1.0)
    plt.plot(results, marker=".")
    plt.legend(tuple(results.columns.values), loc=2)
    plt.savefig(plot_path(metric))
    plt.close()


def conflict_uncertainty_scatter(results, plot_path):
    """
    Create scatterplots of conflict vs uncertainty to be able to identify regions where the algorithm is more/less
    successful.
    @param results: A pandas dataframe with one column for each interesting cutoff. Each "1" in a column will be plotted
    as one cross in the scatterplot for this column, "0" values are ignored.
    @param plot_path: A function that can take a local file name and give back the full path to that file
    @return: None
    """

    #one scatterplot for each columnt
    for cutoff in results.columns:
        r = results[results[cutoff] == 1]
        conflict, uncertainty = zip(*r.index.tolist())
        plt.plot(conflict, uncertainty, 'x', color="#6dad22")
        plt.xticks([0.0, 0.5, 1.0], [0.0, 0.5, 1.0], fontsize=18)
        plt.yticks([0.5, 1.0], [0.5, 1.0], fontsize=18)
        plt.xlabel("conflict", fontsize=18)
        plt.ylabel("uncertainty", fontsize=18)
        plt.xlim(0, 1.0)
        plt.ylim(0, 1.0)
        f = plot_path(cutoff)
        plt.savefig(plot_path(cutoff))
        plt.close()


def comparison_histogram(results, plot_path):
    """
    Compare results (typically true positives) using a bar plot with one set of bars for every possible user action.
    The resulting plot allows to easier see for which actions an algorithm/algorithm setting is more successful.
    @param results: A pandas dataframe with one row for every user action and one column for every algorithm/setting to
    be compared.
    @param plot_path:
    @return:
    """

    results.plot(kind="bar", colormap="Greens")
    plt.subplots_adjust(bottom=0.4)
    plt.savefig(plot_path("hist"))


def plot_observations(source, observations, plot_path, plot_best=5):
    """
    Show which actions typically follow a given user action.
    @param source: The name of the last user action.
    @param observations: A pandas dataframe with one column for each following user action and one row for each bin
    used by the classifier. Each item in the matrix describes how often a user action has been observed in each bin.
    @param plot_path: function that can take a local file name and give back the full path to that file,
    @param plot_best: Only the plot-best number of actions with the most total number of observations in these bins will
    be shown in the resulting plot.
    @return:
    """
    #keep only actions with highest numbers of observations
    observations_per_action = observations.sum()
    observations_per_action.sort(ascending=False)
    most_observations = observations_per_action.index[0:plot_best]
    observations = observations[most_observations]

    #perform the plotting
    plt.figure()
    observations.plot()
    plt.xlabel("Time since setting changed [seconds]")
    plt.ylabel("Observed actions (smoothed)")
    plt.savefig(plot_path(source))