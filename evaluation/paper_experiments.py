# -*- coding: UTF-8 -*-
"""
This file contains experiments for evaluating the proposed recommender system. The experiments are explained
in more detail in:
- the accompanying paper (add link when it article is published online)
- my dissertation (http://www.diva-portal.org/smash/record.jsf?pid=diva2:650328)
Each method also contains references to the relevant sections in paper/dissertation.
"""

import timeit
import os

import numpy
import pandas

from evaluation.experiment import Experiment
from evaluation.metrics import results_as_dataframe, quality_metrics
from evaluation import plot
from classifiers.randomc import RandomClassifier
from classifiers.bayes import NaiveBayesClassifier
from classifiers.temporal import TemporalEvidencesClassifier, configure_dynamic_cutoff
from classifiers.binning import initialize_bins


#per default store plots in ../plots/
default_plot_directory = os.path.join(os.pardir, "plots")

class PaperExperiments():

    def __init__(self, data, cutoff_results_at=None, plot_directory=default_plot_directory, img_type="pdf"):
        """
        Initialize the experimental framework.
        @param data: The dataset on which to test the classifiers.
        @param cutoff_results_at: Cut the recommendation list at cutoff_results_at.  At any given time only a limited
        number of services can be available and can be recommended, e.g. for 10 binary sensors, 10 services are typically
        available. The only anomaly is right at the beginning of the dataset, where the current status of a sensor is not
        known, in this case more than 10 services can be recommended. However, there will be very few instances where
        this is the case and recommendation results can therefore be statistically insignificant.
        @param plot_directory: Base directory in which to store plots.
        @param img_type: Image type for the plots.
        @return:
        """
        self.data = data
        self.plot_directory = plot_directory
        self.img_type = img_type
        self.cutoff_results_at = cutoff_results_at

    def compare_classifiers(self):
        """
        Compare quality and runtimes of several classifiers for one dataset. Performs 10-fold cross-validation. Details
        for this experiment can be found in the paper in Section 6.4 and in the dissertation in Section 5.5.4,
        """
        print "Compare classifiers for dataset " + self.data.name
        experiment = Experiment(self.data)
        experiment.add_classifier(TemporalEvidencesClassifier(self.data.features, self.data.target_names), name="Our method")
        experiment.add_classifier(NaiveBayesClassifier(self.data.features, self.data.target_names), name="Naive Bayes")
        experiment.add_classifier(RandomClassifier(self.data.features, self.data.target_names), name="Random")
        results = experiment.run(folds=10)

        results.print_quality_comparison_at_cutoff(cutoff=1, metrics=["Recall", "Precision", "F1"])
        results.print_runtime_comparison()

        plot_conf = plot.plot_config(self.plot_directory, sub_dirs=[self.data.name], img_type=self.img_type)
        results.plot_quality_comparison(metrics=["Recall", "Precision", "F1"], plot_config=plot_conf,
                                        cutoff_results_at=self.cutoff_results_at)


    def evaluate_interval_settings(self):
        """
        Evaluate how the proposed recommender system copes with varying interval widths and Delta t_max settings. Details
        for this experiment can be found in the paper in Section 6.5 and in the Dissertation in Section 5.5.6.
        There was a bug in the original code that lead to wrong results when there was only one interval in the list of
        bins (case "Delta t_max=10s").
        """
        print "Comparing different interval settings"
        experiment = Experiment(self.data)

        intervals_to_test = [#test various settings for delta t_max
                             ("Delta t_max=1200s", initialize_bins(start=0, end=60, width=10) +
                                                   initialize_bins(start=60, end=1200, width=30)),
                             ("Delta t_max=120s",  initialize_bins(start=0, end=60, width=10) +
                                                   initialize_bins(start=60, end=120, width=30)),
                             ("Delta t_max=60s",   initialize_bins(start=0, end=60, width=10)),
                             ("Delta t_max=30s",   initialize_bins(start=0, end=30, width=10)),
                             ("Delta t_max=10s",   initialize_bins(start=0, end=10, width=10)),
                             #test various interval widths
                             ("all intervals 2s wide",   initialize_bins(start=0, end=300, width=2)),
                             ("all intervals 4s wide",   initialize_bins(start=0, end=300, width=4)),
                             ("all intervals 6s wide",   initialize_bins(start=0, end=300, width=6)),
                             ("all intervals 8s wide",   initialize_bins(start=0, end=300, width=8)),
                             ("all intervals 30s wide",  initialize_bins(start=0, end=300, width=30)),
                             ("all intervals 50s wide",  initialize_bins(start=0, end=300, width=50)),
                             ("all intervals 100s wide", initialize_bins(start=0, end=300, width=100))]

        for (name, bins) in intervals_to_test:
            experiment.add_classifier(TemporalEvidencesClassifier(self.data.features, self.data.target_names,
                                      bins=bins), name=name)

        results = experiment.run(folds=10)
        results.print_quality_comparison_at_cutoff(cutoff=1, metrics=["Recall", "Precision", "F1"])

    def scatter_conflict_uncertainty(self):
        """
        Scatter conflict versus uncertainty at different cutoffs to find regions of uncertainty/conflict where the algorithm
        is more/less successful. Further details for this experiment can be found in the paper in Section 6.6 and the
        dissertation in Section 5.5.7.
        @return:
        """

        #run the classifier on the whole dataset
        cls = TemporalEvidencesClassifier(self.data.features, self.data.target_names)
        cls = cls.fit(self.data.data, self.data.target)
        results = cls.predict(self.data.data, include_conflict_theta=True)

        #extract conflict and uncertainty and convert recommendations to pandas representation
        recommendations, conflict, uncertainty = zip(*results)
        results = results_as_dataframe(self.data.target, list(recommendations))

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
        conf = plot.plot_config(self.plot_directory, sub_dirs=[self.data.name, "scatter"],
                                prefix="found_within_", img_type=self.img_type)
        plot.conflict_uncertainty_scatter(found_within, conf)

        #not found withing: the correct service was not found within X recommendations, is the reverse of found_within
        not_found_within = found_within.apply(lambda col: 1-col)
        #create one plot for each cutoff
        conf = plot.plot_config(self.plot_directory, sub_dirs=[self.data.name, "scatter"],
                                prefix="not_found_within_", img_type=self.img_type)
        plot.conflict_uncertainty_scatter(not_found_within, conf)


    def evaluate_dynamic_cutoff(self):
        """
        Evaluates the benefit of dynamic cutoff methods, i.e. show less recommendations if uncertainty and conflict are low.
        Further details for this experiment can be found in the paper in Section 6.6 and the dissertation in Section 5.5.7
        """
        print "Evaluating use of dynamic cutoff methods"
        experiment = Experiment(self.data)
        methods_to_test = [("Fixed cutoff", None),
                           ("dynamic cutoff=4", configure_dynamic_cutoff(1.0, 0.4, 4)),
                           ("dynamic cutoff=2", configure_dynamic_cutoff(1.0, 0.4, 2))]

        for name, method in methods_to_test:
            experiment.add_classifier(TemporalEvidencesClassifier(self.data.features, self.data.target_names,
                                      postprocess=method), name=name)
        results = experiment.run(folds=10)

        pandas.set_option('expand_frame_repr', False)
        pandas.set_option('max_columns', 4)
        print "Maximum 5 recommendations"
        results.print_quality_comparison_at_cutoff(cutoff=5, metrics=quality_metrics)
        print "Maximum 10 recommendations"
        results.print_quality_comparison_at_cutoff(cutoff=10, metrics=quality_metrics)

    def evaluate_training_size(self):
        """
        Evaluates how the classifiers behave for different sizes of the training dataset. Further details for this
        experiment can be found in the dissertation in Section 5.5.5
        """

        elapsed_time = lambda end: end - self.data.times[0]
        timedelta_to_seconds = lambda td: int(td.item()/(1000.0*1000.0*1000.0))
        seconds_to_days = lambda seconds: seconds//float(24*60*60)
        elapsed_time_days = lambda end: seconds_to_days(timedelta_to_seconds(elapsed_time(end)))

        def divide_dataset():
            #how many items in training dataset
            dataset_size = len(self.data.data)
            train_sizes = [10, 25, 50, 75] + [int(r*dataset_size) for r in list(numpy.arange(0.05, 1.00, 0.05))]
            #how much time (in days) is covered by these items
            train_times = [elapsed_time_days(self.data.times[train_size]) for train_size in train_sizes]
            #for each training size, create a training dataset that contains only the first train_size items
            train_datasets = [numpy.array([True]*train_size + [False]*(dataset_size-train_size))
                              for train_size in train_sizes]
            #test data is always the whole dataset
            test_datasets = [numpy.array([True]*dataset_size) for train_size in train_sizes]
            return train_sizes, train_times, train_datasets, test_datasets

        def initialize_experiment():
            experiment = Experiment(self.data)
            experiment.add_classifier(TemporalEvidencesClassifier(self.data.features, self.data.target_names),
                                      name="Our method")
            experiment.add_classifier(NaiveBayesClassifier(self.data.features, self.data.target_names),
                                      name="Naive Bayes")
            return experiment


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
            r.name = metric
            print metric
            print r

        #plot means for interesting metrics
        plot_conf = plot.plot_config(self.plot_directory, sub_dirs=[self.data.name], prefix="trainsize_",
                                     img_type=self.img_type)
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
from evaluation.synthetic import generate_trained_classifier
from evaluation.paper_experiments import num_sensors, nominal_values_per_sensor, num_instances
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
    #print "Total testing time %.4f [ms]" %test_time
    print "Testing time per instance %.4f [ms]" % test_time_per_instance



