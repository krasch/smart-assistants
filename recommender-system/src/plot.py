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


def plot_evidences(classifier):
    bins=classifier.binning_method.bins
    for source in classifier.sources.values():
        observations_by_target=dict()
        for t,target in enumerate(classifier.target_names):
            observations_by_target[target]=[source.temporal_counts[b][t] for b in range(len(bins))]
        observed_sorted=sorted(observations_by_target,key=lambda k:sum(observations_by_target[k]),reverse=True)[0:5]  #sort by sum of counts
        for observed in observed_sorted:
            plt.plot(bins,observations_by_target[observed])
        plt.xlabel("Time since setting changed [seconds]")
        plt.ylabel("Observed actions (smoothed)")
        plt.legend(observed_sorted) 
        plt.savefig("../plots/"+source.source_name()+".pdf")
        plt.close()   


def plot_quality_comparison(results, metric, full_plot_path):
    """
    Create a lineplot, with one line for every evaluated classifier, showing the measured metric for this classifier.
    @param results: A pandas dataframe with one column for every classifier, containing measurements for the metric.
    @param metric: The metric to plot.
    @param full_plot_path: A function that can take a local file name and give back the full path to that file,
    @return: None
    """
    plt.figure(figsize=(6, 6), dpi=300)
    plt.ylabel(metric, fontsize=18)
    plt.xlabel('recommendation cutoff', fontsize=18)
    if not metric == "# of recommendations":
        plt.ylim(0.0, 1.0)
    plt.plot(results, marker=".")
    plt.legend(tuple(results.columns.values), loc=4)
    plt.savefig(full_plot_path(metric))
    plt.close()


def plot_train_size(results, metric, full_plot_path):
    """
    Create a lineplot, that shows how training size influences the given metric, with one line for every evaluated
    classifier.
    @param results: A pandas dataframe with one column for every classifier, containing measurements for the metric. The
    dataframe should have a multiIndex consisting of a tuple (size of training data, elapsed time).
    @param metric: The name of the metric that is to plot.
    @param full_plot_path:  A function that can take a local file name and give back the full path to that file.
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
    plt.savefig(full_plot_path(metric))
    plt.close()


def conflict_uncertainty_scatter(results, full_plot_path):
    """
    Create scatterplots of conflict vs uncertainty to be able to identify regions where the algorithm is more/less
    successful.
    @param results: A pandas dataframe with one column for each interesting cutoff. Each "1" in a column will be plotted
    as one cross in the scatterplot for this column, "0" values are ignored.
    @param full_plot_path: A function that can take a local file name and give back the full path to that file
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
        f = full_plot_path(cutoff)
        plt.savefig(full_plot_path(cutoff))
        plt.close()


def histogram(to_compare,data,filename):
    bar_width = 0.75       
    colors=["lightblue","lightgreen","lightcoral","orange","pink"]
    services=[service for (service,executed,correct) in data[to_compare[0]]]

    for (c_index,c) in enumerate(to_compare):
        x=range(c_index,len(services)*(len(to_compare)+1),len(to_compare)+1)  
        y=[executed for (service,executed,correct) in data[c]]       
        plt.bar(x,y,bar_width,color='grey',alpha=0.3)
        y=[correct for (service,executed,correct) in data[c]]       
        plt.bar(x,y,bar_width,color=colors[c_index])
    
    x=range(int(len(to_compare)/2),len(services)*(len(to_compare)+1),len(to_compare)+1)
    plt.xticks(x,services,rotation=90,size=9)
    plt.xlim(0,x[-1]+1) 
    plt.subplots_adjust(bottom=0.4)
    #plt.legend(tuple(to_compare),loc=0) 
    plt.savefig(filename)
    plt.close()   
