from collections import defaultdict

import numpy
import pandas

from data.kasteren import load_scikit as load_kasteren
from experiment import Experiment
from classifiers.randomc import RandomClassifier
from classifiers.bayes import NaiveBayesClassifier
from classifiers.temporal import TemporalEvidencesClassifier
from classifiers.binners import StaticBinning
from classifiers.postprocess import dynamic_cutoff
import plot


"""
This file contains experiments for evaluating the proposed recommender system. The experiments are explained
in more detail in:
- the accompanying paper (todo link)
- my dissertation (http://www.diva-portal.org/smash/record.jsf?pid=diva2:650328)
"""


def compare_classifiers(data):
    """
    Compare quality and runtimes of several classifiers for one dataset. Performs 10-fold cross-validation. Details
    for this experiment can be found in the paper in Section 6.4 and in the dissertation in Section 5.5.4,
    @param data: The dataset on which to run the comparison.
    """
    print "Compare classifiers for dataset " + data.name
    experiment = Experiment(data)
    #experiment.add_classifier(TemporalEvidencesClassifier(data.features, data.target_names), name="Our method")
    #experiment.add_classifier(NaiveBayesClassifier(data.features, data.target_names), name="Naive Bayes")
    experiment.add_classifier(RandomClassifier(data.features, data.target_names), name="Random")
    results = experiment.run(folds=10)
    results.print_runtime_comparison()
    results.print_quality_comparison()
    #experiment.plot_quality_comparison()


def evaluate_interval_settings(data):
    """
    Evaluate how the proposed recommender system copes with varying interval widths and Delta t_max settings. Details
    for this experiment can be found in the paper in Section 6.5 and in the Dissertation in Section 5.5.6.
    """
    print "Comparing different interval settings"
    experiment = Experiment(data)
    bins = lambda start, end, width: list(range(start, end, width))
    intervals_to_test = [#test various settings for delta t_max
                         ("Delta t_max=10s",   bins(start=0, end=10, width=10)),
                         ("Delta t_max=15s",   bins(start=0, end=15, width=10)),
                         ("Delta t_max=30s",   bins(start=0, end=30, width=10)),
                         ("Delta t_max=60s",   bins(start=0, end=60, width=10)),
                         ("Delta t_max=120s",  bins(start=0, end=60, width=10)+bins(start=60, end=120, width=30)),
                         ("Delta t_max=1200s", bins(start=0, end=60, width=10)+bins(start=60, end=1200, width=30)),
                         #test various interval widths
                         ("all intervals 2s wide",   bins(start=0, end=300, width=2)),
                         ("all intervals 4s wide",   bins(start=0, end=300, width=4)),
                         ("all intervals 6s wide",   bins(start=0, end=300, width=6)),
                         ("all intervals 8s wide",   bins(start=0, end=300, width=8)),
                         ("all intervals 30s wide",  bins(start=0, end=300, width=30)),
                         ("all intervals 50s wide",  bins(start=0, end=300, width=50)),
                         ("all intervals 100s wide", bins(start=0, end=300, width=100))]

    for (name, bins) in intervals_to_test:
        experiment.add_classifier(TemporalEvidencesClassifier(data.features, data.target_names,
                                  binning_method=StaticBinning(bins=bins)), name=name)

    results = experiment.run(folds=10)
    results.print_runtime_comparison()
    results.print_quality_comparison_at_cutoff(cutoff=1)


def evaluate_dynamic_cutoff(data):
    """
    Evaluates the benefit of dynamic cutoff methods, i.e. show less recommendations if uncertainty and conflict are low.
    Further details for this experiment can be found in the paper in Section 6.6 and the dissertation in Section 5.5.7
    @param data: The dataset used for evaluation.
    """
    print "Evaluating use of dynamic cutoff methods"
    experiment = Experiment(data)
    methods_to_test = [("Fixed cutoff", None),
                       ("dynamic cutoff=4", dynamic_cutoff(1.0, 0.4, 4)),
                       ("dynamic cutoff=2", dynamic_cutoff(1.0, 0.4, 2))]
    for name, method in methods_to_test:
        experiment.add_classifier(TemporalEvidencesClassifier(data.features, data.target_names,
                                  postprocess=method), name=name)
    experiment.run(folds=10)
    experiment.print_quality_comparison()



def evaluate_training_size(data):
    """
    Evaluates how the classifiers behave for different sizes of the training dataset. Further details for this
    experiment can be found in the dissertation in Section 5.5.5
    @param data: The dataset used for evaluation.
    """

    elapsed_time_seconds = lambda end: (end - data.times[0]).days*24*60*60 + (end-data.times[0]).seconds
    elapsed_time_days = lambda end: elapsed_time_seconds(end)/float(24*60*60)

    def divide_dataset():
        #how many items in training dataset
        dataset_size = len(data.data)
        train_sizes = [10, 25, 50, 75] + [int(r*dataset_size) for r in list(numpy.arange(0.05, 1.00, 0.05))]
        #how much time (in days) is covered by these items
        train_times = [elapsed_time_days(data.times[train_size]) for train_size in train_sizes]
        #for each training size, create a training dataset that contains only the first train_size items
        train_datasets = [numpy.array([True]*train_size + [False]*(dataset_size-train_size))
                          for train_size in train_sizes]
        #test data is always the whole dataset
        test_datasets = [numpy.array([True]*dataset_size) for train_size in train_sizes]
        return train_sizes, train_times, train_datasets, test_datasets

    def initialize_experiment():
        experiment = Experiment(data)
        experiment.add_classifier(TemporalEvidencesClassifier(dataset.features, dataset.target_names), name="Our method")
        experiment.add_classifier(NaiveBayesClassifier(data.features, data.target_names), name="Naive Bayes")
        experiment.add_classifier(RandomClassifier(data.features, data.target_names), name="Random")
        return experiment

    def add_index_to_results(res, sizes, times):
        df = pandas.concat(res, axis=0)
        df.index = pandas.MultiIndex.from_tuples(zip(sizes, times),
                                                 names=["Size of dataset", "Elapsed time (days)"])
        return df

    #the classifiers will be trained with increasingly larger training datasets, create those here
    train_sizes, train_times, train_datasets, test_datasets = divide_dataset()

    #run the experiment for each of thee created datasets
    results = defaultdict(list)
    for train_data, test_data in zip(train_datasets, test_datasets):
        experiment = initialize_experiment()
        #run only one fold for this data for each of the classifiers
        for cls in experiment.classifiers:
            experiment.run_with_classifier(cls, [(train_data, test_data)])
        #store results for cutoff=1
        for metric in experiment.quality_stats:
            results[metric].append(experiment.compare_quality_at_cutoff(metric, "Mean", cutoff=1).transpose())

    #add multi-index of training sizes and training times to the results
    results = {metric: add_index_to_results(result, train_sizes, train_times) for metric, result in results.items()}

    #print out results
    for metric in experiment.quality_stats:
        print metric
        print results[metric]


dataset = load_kasteren("houseA")
compare_classifiers(dataset)
#evaluate_interval_settings(dataset)
#evaluate_dynamic_cutoff(dataset)
#evaluate_training_size(dataset)
