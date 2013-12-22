from datetime import datetime
from collections import defaultdict
from math import sqrt

from sklearn.cross_validation import KFold
import numpy
from scipy import stats as scipy_stats
import pandas

from classifiers import metrics
import plot

calculated_stats = ["Mean", "Std deviation", "Confidence interval"]
runtime_metrics = ["Training time", "Overall testing time", "Individual testing time"]
accuracy_metrics = ["Recall", "Precision", "F1", "# of recommendations"]


def confidence_interval(data, alpha=0.1):
    """
    Calculate the confidence interval for each column in a pandas dataframe.
    @param data: A pandas dataframe with one or several columns.
    @param alpha: The confidence level, by default the 90% confidence interval is calculated.
    @return: A time series where each entry contains the confidence-interval for the corresponding column.
    """
    alpha = 0.1
    t = lambda column: scipy_stats.t.isf(alpha/2.0, len(column)-1)
    width = lambda column: t(column) * numpy.std(column.values, ddof=1)/sqrt(len(column))
    formatted_interval = lambda column: "%.2f +/- %.4f" % (column.mean(), width(column))
    return pandas.TimeSeries([formatted_interval(data[c]) for c in data.columns])


class RuntimeResults():
    """
    Store all measurements of training and testing runtimes for one classifier.
    """
    def __init__(self):
        self.train_times = []
        self.test_times = []
        self.test_times_ind = []

    def add_measurements(self, train_time, test_time, test_time_ind):
        self.train_times.append(train_time)
        self.test_times.append(test_time)
        self.test_times_ind.append(test_time_ind)

    def get_stats(self):
        """
        Calculate overall statistics for all measured runtime metrics.
        @return: A dictionary that contains statistics for each runtime metric. Each statistic object is a pandas
                 dataframe with only one row and three columns: mean, standard deviation and confidence interval.
        """
        def calculate_stats(measurements):
            df = pandas.DataFrame(measurements)
            return pandas.DataFrame([df.mean(), df.std(), confidence_interval(df)],
                                    index=calculated_stats).transpose()

        return {"Training time": calculate_stats(self.train_times),
                "Overall testing time": calculate_stats(self.test_times),
                "Individual testing time": calculate_stats(self.test_times_ind)}


class AccuracyResults():

    """
    Store all measurements of recommendation accuracy for one classifier.
    """
    def __init__(self):
        self.precision = []
        self.recall = []
        self.f1 = []
        """ in some experiments we restrict the number of recommendations shown to the user,
            not strictly an accuracy measurement, but data fits here """
        self.num_recommendations = []

    def add_measurements(self, (precision, recall, f1), num_recommendations):
        self.precision.append(precision)
        self.recall.append(recall)
        self.f1.append(f1)
        self.num_recommendations.append(num_recommendations)

    def get_stats(self):
        """
        Calculate overall statistics for all measured accuracy metrics.
        @return: A dictionary that contains statistics for each accuracy metric. Each statistics object is a pandas
        dataframe with several rows, each row corresponds to one cutoff setting (e.g. show only the best recommendation,
        show two recommendations etc.). The dataframe has three columns: mean, standard deviation, confidence interval.
        """
        def calculate_stats(measurements):
            df = pandas.DataFrame(measurements)
            stats = pandas.DataFrame([df.mean(), df.std(), confidence_interval(df)],
                                     index=calculated_stats).transpose()
            stats.index.name = "cutoff"
            return stats

        return {"Precision": calculate_stats(self.precision),
                "Recall": calculate_stats(self.recall),
                "F1": calculate_stats(self.f1),
                "# of recommendations": calculate_stats(self.num_recommendations)}


class Experiment:

    def __init__(self, dataset):
        self.dataset = dataset
        self.classifiers = []
        self.accuracy_stats = defaultdict(dict)
        self.runtime_stats = defaultdict(dict)
  
    def add_classifier(self, cls, name=None):
        if name is None:
            name = cls.name
        self.classifiers.append((name, cls))

    def __run_with_classifier__(self, data_for_folds, cls):
        """
        @param data_for_folds: Contains one list of True/False values for each of the folds to be run. Each list states
                               for every item of the dataset, whether the item is in the current fold part of the
                               training dataset or the test dataset.
        @param cls: Classifier to use in the experiment.
        @return: Measurements for accuracy and runtime metrics.
        """
        runtimes = RuntimeResults()
        accuracy = AccuracyResults()
        for train, test in data_for_folds:
            #get the training and testing data for this fold
            data_train, data_test = self.dataset.data[train], self.dataset.data[test]
            target_train, target_test = self.dataset.target[train], self.dataset.target[test]
            #performing training
            train_time = datetime.now()
            cls = cls.fit(data_train, target_train)
            train_time = (datetime.now()-train_time).microseconds/1000.0
            #apply the classifier on the test data
            test_time = datetime.now()
            predicted = cls.predict(data_test)
            test_time = (datetime.now()-test_time).microseconds/1000.0
            #add measurements to results for this classifier
            runtimes.add_measurements(train_time, test_time, test_time/float(len(data_test)))
            accuracy.add_measurements(metrics.multiple_predictions_scores(target_test, predicted, 20),
                                      metrics.num_predictions(target_test, predicted, 20))
        return accuracy, runtimes

    def run(self, folds=10):
        """
        Run the experiment with all classifiers.
        @param folds: How many folds to run, perform 10-fold cross validation by default. folds must be >=2
        """
        assert(folds >= 2)
        data_for_folds = KFold(len(self.dataset.data), n_folds=folds, indices=False)
        for name, cls in self.classifiers:
            #run simulation for one classifier
            accuracy,runtimes = self.__run_with_classifier__(data_for_folds, cls)
            #sort results by metric
            for metric, stats in accuracy.get_stats().items():
                self.accuracy_stats[metric][name] = stats
            for metric, stats in runtimes.get_stats().items():
                self.runtime_stats[metric][name] = stats

    def __compare_classifiers__(self, metric, statistic):
        """
        Grab results for given metric and statistic for all tested classifiers.
        @param metric: Name of one of the runtime or one of the accuracy metrics.
        @param statistic: Which statistic to compare (Mean, Standard deviation, Confidence interval)
        @return: A pandas dataframe with one row for every classifier and one column listing the calculated statistic
                 for the given metric for each of the classifiers.
        """
        assert(statistic in calculated_stats)
        assert(metric in self.runtime_stats or metric in self.accuracy_stats)
        if metric in self.runtime_stats:
            stats = self.runtime_stats[metric]
        else:
            stats = self.accuracy_stats[metric]
        return pandas.DataFrame([stats[name][statistic] for name, cls in self.classifiers],
                                index=[name for name, cls in self.classifiers]).transpose()

    def print_accuracy_comparison(self):
        for metric in accuracy_metrics:
            print "Results for %s"%metric
            print self.__compare_classifiers__(metric, "Confidence interval")

    def print_runtime_comparison(self):
        comparison = pandas.concat([self.__compare_classifiers__(metric, "Confidence interval")
                            for metric in runtime_metrics]).transpose()
        comparison.columns = runtime_metrics
        print comparison

    def plot_accuracy_comparison(self):
        for metric in accuracy_metrics:
            results = self.__compare_classifiers__(metric, "Mean")
            plot.plot_accuracy_comparison(results, metric, "../plots/houseA/%s.pdf" % metric)


