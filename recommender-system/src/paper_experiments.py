import os
import timeit

import numpy
import pandas

from dataset import load_dataset_as_sklearn
from experiment.experiment import Experiment
from experiment.metrics import results_as_dataframe
from experiment import plot
from classifiers.randomc import RandomClassifier
from classifiers.bayes import NaiveBayesClassifier
from classifiers.temporal import TemporalEvidencesClassifier
from classifiers.binners import StaticBinning
from classifiers.postprocess import dynamic_cutoff


"""
This file contains experiments for evaluating the proposed recommender system. The experiments are explained
in more detail in:
- the accompanying paper (todo link)
- my dissertation (http://www.diva-portal.org/smash/record.jsf?pid=diva2:650328)
Each method also contains references to the relevant sections in paper/dissertation.
"""


#store plots in ../plots/
plot_directory = os.path.join(os.pardir, "plots")
img_type = "pdf"


def compare_classifiers(data):
    """
    Compare quality and runtimes of several classifiers for one dataset. Performs 10-fold cross-validation. Details
    for this experiment can be found in the paper in Section 6.4 and in the dissertation in Section 5.5.4,
    @param data: The dataset on which to run the comparison.
    """
    print "Compare classifiers for dataset " + data.name
    experiment = Experiment(data)
    experiment.add_classifier(TemporalEvidencesClassifier(data.features, data.target_names), name="Our method")
    experiment.add_classifier(NaiveBayesClassifier(data.features, data.target_names), name="Naive Bayes")
    experiment.add_classifier(RandomClassifier(data.features, data.target_names), name="Random")
    results = experiment.run(folds=10)

    results.print_runtime_comparison()
    results.print_quality_comparison()

    plot_conf = plot.plot_config(plot_directory, sub_dirs=[data.name], img_type=img_type)
    results.plot_quality_comparison(plot_conf)


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


def scatter_conflict_uncertainty(data):
    """
    Scatter conflict versus uncertainty at different cutoffs to find regions of uncertainty/conflict where the algorithm
    is more/less successful. Further details for this experiment can be found in the paper in Section 6.6 and the
    dissertation in Section 5.5.7.
    @param data: The dataset used for the evaluation
    @return:
    """

    #run the classifier on the whole dataset
    cls = TemporalEvidencesClassifier(data.features, data.target_names)
    cls = cls.fit(data.data, data.target)
    results = cls.predict(data.data, include_conflict_theta=True)

    #extract conflict and uncertainty and convert recommendations to pandas representation
    recommendations, conflict, uncertainty = zip(*results)
    results = results_as_dataframe(data.target, list(recommendations))

    #for each row, mark correct recommendations with "1", false recommendations with "0"
    find_matches_in_row = lambda row: [1 if col == row.name else 0 for col in row]
    results = results.apply(find_matches_in_row, axis=1)

    #set uncertainty and conflict as multi-index
    results.index = pandas.MultiIndex.from_tuples(zip(conflict, uncertainty),
                                                  names=["Conflict", "Uncertainty"])

    #found_within: the correct service was found within X recommendations
    #-> apply cumulative sum on each row so that the "1" marker is set for all columns after it first appears
    found_within = results.cumsum(axis=1)
    #create one plot for each cutoff
    conf = plot.plot_config(plot_directory, sub_dirs=[data.name, "scatter"],
                            prefix="found_within_", img_type=img_type)
    plot.conflict_uncertainty_scatter(found_within, conf)

    #not found withing: the correct service was not found within X recommendations, is the reverse of found_within
    not_found_within = found_within.apply(lambda col: 1-col)
    #create one plot for each cutoff
    conf = plot.plot_config(plot_directory, sub_dirs=[data.name, "scatter"],
                            prefix="not_found_within_", img_type=img_type)
    plot.conflict_uncertainty_scatter(not_found_within, conf)


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
    results = experiment.run(folds=10)
    results.print_quality_comparison()


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
        #experiment.add_classifier(TemporalEvidencesClassifier(dataset.features, dataset.target_names), name="Our method")
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

    #run the experiment for each of the created datasets
    results = []
    for train_data, test_data in zip(train_datasets, test_datasets):
        experiment = initialize_experiment()
        #run with all defined classifiers
        stats = [experiment.run_with_classifier(cls, [(train_data, test_data)])
                 for cls in experiment.classifiers]
        #combine results of all classifiers for this training dataset, keep only results for cutoff=1
        quality_stats = pandas.concat([quality for quality, runtime in stats], axis=1).loc[1]
        results.append(quality_stats)


    #make one big matrix with all results and add multi-index of training sizes and training times
    results = pandas.concat(results, axis=1).transpose()
    results.index = pandas.MultiIndex.from_tuples(zip(train_sizes, train_times),
                                                 names=["Size of dataset", "Elapsed time (days)"])

    #print confidence intervals for interesting metrics
    interesting_columns = lambda metric: [(cls.name,metric, "Confidence interval") for cls in experiment.classifiers]
    for metric in ["Precision", "Recall", "F1"]:
        r = results[interesting_columns(metric)]
        r.columns = [cls.name for cls in experiment.classifiers]
        print metric
        print r

    #plot means for interesting metrics
    plot_conf = plot.plot_config(plot_directory, sub_dirs=[data.name], prefix="trainsize_", img_type=img_type)
    interesting_columns = lambda metric: [(cls.name, metric, "Mean") for cls in experiment.classifiers]
    for metric in ["Precision", "Recall", "F1"]:
        r = results[interesting_columns(metric)]
        r.columns = [cls.name for cls in experiment.classifiers]
        plot.plot_train_size(r, metric, plot_conf)


def scalability_experiment():
    """
    Evaluate how the proposed recommendation algorithm scales for larger datasets with many sensors and several nominal
    values per sensor. The data for this experiment is synthetically generated. Details for the experiment can be found
    in the paper in Section 6.7 and in the dissertation in Section 5.5.9,
    @return:
    """

    #setup necessary to run timeit function
    setup = '''
from experiment.synthetic import generate_trained_classifier
from paper_experiments import num_sensors, nominal_values_per_sensor, num_instances
cls, test_data = generate_trained_classifier(num_sensors=num_sensors,\
                                             nominal_values_per_sensor=nominal_values_per_sensor, \
                                             num_test_instances=num_instances)
'''
    #evaluation parameters
    global num_instances, num_sensors, nominal_values_per_sensor
    num_instances = 1000
    num_sensors = 100
    nominal_values_per_sensor = 5
    seconds_to_milliseconds = lambda seconds: seconds*1000.0

    #evaluate
    timer = timeit.Timer('cls.predict(test_data.data)', setup=setup)
    test_time = seconds_to_milliseconds(min(timer.repeat(repeat=3, number=1)))
    test_time_per_instance = test_time / num_instances
    print "Testing time per instance %.4f [ms]" % test_time_per_instance



dataset = load_dataset_as_sklearn("../datasets/houseA.csv", "../datasets/houseA.config")
compare_classifiers(dataset)
#evaluate_interval_settings(dataset)
#scatter_conflict_uncertainty(dataset)
#evaluate_dynamic_cutoff(dataset)
#evaluate_training_size(dataset)
#scalability_experiment()