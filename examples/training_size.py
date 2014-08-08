# -*- coding: UTF-8 -*-
"""
Evaluates how the classifiers behave for different sizes of the training dataset. Further details for this
experiment can be found in the dissertation in Section 5.5.5
"""
import sys
sys.path.append("..") 

import pandas
import numpy

from evaluation.experiment import Experiment
from evaluation import plot
from recsys.classifiers.temporal import TemporalEvidencesClassifier
from recsys.classifiers.bayes import NaiveBayesClassifier
from recsys.dataset import load_dataset
import config


#configuration
data = load_dataset("../datasets/houseA.csv", "../datasets/houseA.config")

#some util methods
elapsed_time = lambda end: end - data.times[0]
timedelta_to_seconds = lambda td: int(td.item()/(1000.0*1000.0*1000.0))
seconds_to_days = lambda seconds: seconds//float(24*60*60)
elapsed_time_days = lambda end: seconds_to_days(timedelta_to_seconds(elapsed_time(end)))

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
    experiment.add_classifier(TemporalEvidencesClassifier(data.features, data.target_names),
                              name="Our method")
    experiment.add_classifier(NaiveBayesClassifier(data.features, data.target_names),
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
plot_conf = plot.plot_config(config.plot_directory, sub_dirs=[data.name], prefix="trainsize_",
                             img_type=config.img_type)
interesting_columns = lambda metric: [(cls.name, metric, "Mean") for cls in experiment.classifiers]
for metric in ["Precision", "Recall", "F1"]:
    r = results[interesting_columns(metric)]
    r.columns = [cls.name for cls in experiment.classifiers]
    plot.plot_train_size(r, metric, plot_conf)
