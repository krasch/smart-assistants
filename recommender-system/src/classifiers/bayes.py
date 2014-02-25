# -*- coding: UTF-8 -*-

import pandas
import numpy

from base import BaseClassifier


class NaiveBayesClassifier(BaseClassifier):
    """
    Implements a standard Naive Bayes classifier.
    """

    name = "NaiveBayes"

    def __init__(self, features, target_names):
        BaseClassifier.__init__(self, features, target_names)

    def fit(self, train_data,train_target):
        """
        Train the classifier.
        @param train_data: A matrix with len(self.features) columns and one row for each instance in the dataset. Each
        row describes a user situation with current sensor settings and information on how long these settings have
        not changed.
        @param train_target: An array of targets (length of the array corresponds to the number of rows in train_data).
        Each target represents the action that user performed in the situation described by the corresponding row
        in train_data.
        @return: self-reference for this classifier
        """
        def calculate_priors(data):
            """
            Count how often each target (user action) was seen overall. Performs additive smoothing and normalizes
            the counts.
            @param data:
            @return:
            """
            #count how often each target was seen, set to 0 if target has never been seen
            counts_per_target = data.index.to_series().value_counts(sort=False)
            counts_per_target = counts_per_target.reindex(self.target_names).fillna(0)
            #additive smoothing (add one to every count), necessary so that NaiveBayes does not degrade for zero-counts
            counts_per_target += 1
            #normalize
            normalized_counts = counts_per_target.div(counts_per_target.sum())
            return normalized_counts.values

        def calculate_counts_per_setting(data):
            """
            Count how often each target was seen in each setting.  Performs additive smoothing and normalizes
            the counts.
            @param data:
            @return:
            """
            #count how often each target was seen in each settings, set to 0 if target has never been seen in a setting
            counts_per_setting = data.groupby(data.index, sort=False).sum()
            counts_per_setting = counts_per_setting.reindex(self.target_names).fillna(0)
            #additive smoothing (add one to every count), necessary so that NaiveBayes does not degrade for zero-counts
            counts_per_setting += 1
            #normalize the counts per sensor
            normalize = lambda counts_per_sensor: counts_per_sensor.div(counts_per_sensor.sum(axis=1), axis=0)
            normalized_counts = counts_per_setting.groupby(lambda (sensor, value): sensor, axis=1).transform(normalize)

            #convert to dictionary of numpy arrays for faster calculations later on
            normalized_counts = {setting: normalized_counts[setting].values for setting in normalized_counts.columns}
            return normalized_counts

        #load training data and targets into pandas dataframe
        train_data = pandas.DataFrame(train_data)
        train_data.columns = self.features
        train_data.index = train_target

        #keep only the columns with current sensor settings, since Naive Bayes does not use timedeltas
        train_data = train_data[self.settings_columns]

        #calculate how often each target was seen and how often it was seen in specific settings
        self.priors = calculate_priors(train_data)
        self.counts = calculate_counts_per_setting(train_data)

        return self

    def predict(self, test_data):
        """
        Calculate recommendations for the test_data
        @param test_data: A matrix with len(self.features) columns and one row for each instance in the dataset. Each
        row describes a user situation with current sensor settings and information on how long these settings have
        not changed.
        @return: Resulting recommendations for each instance in the dataset (a list of list of strings).
        """

        #load test data into pandas dataframe
        test_data = pandas.DataFrame(test_data)
        test_data.columns = self.features

        #keep only the columns with current sensor settings, since Naive Bayes does not use timedeltas
        test_data = test_data[self.settings_columns].values

        #calculate sorted recommendations for one instance
        def predict_for_instance(instance):

            #find which sensor values are currently set
            currently_set = self.currently_set(instance)

            #calculate which targets (user actions) are currently possible (possible targets are represented by 1,
            #not currently possible targets are represented by 0)
            possible_targets_mask = self.possible_targets_mask(currently_set)

            #lookup observations for the current settings
            counts = [self.counts[setting] for setting in currently_set]

            #calculate posteriors, set posteriors for not currently possible targets to zero and normalize
            posteriors = reduce(numpy.multiply, counts) * self.priors
            posteriors = posteriors * possible_targets_mask
            normalized_posteriors = posteriors / posteriors.sum()

            #map resulting posteriors to the possible targets and sort
            recommendations = {target: posterior for target, posterior, target_is_possible
                               in zip(self.target_names, normalized_posteriors, possible_targets_mask)
                               if target_is_possible}
            sorted_recommendations = sorted(recommendations, key=recommendations.get, reverse=True)

            return sorted_recommendations

        #calculate recommendations for every instance in the test dataset
        results = [predict_for_instance(test_data[i]) for i in range(len(test_data))]

        return results

    def print_counts_and_priors(self):
        """
        Simple debugging method that prints out the calculated counts and priors after the classifier has been trained.
        @return:
        """
        line_to_string = lambda target, count: "%s %.2f" % (target, count)
        print "\n".join([line_to_string(target, count) for target, count in self.priors.iteritems()])

        format_line = lambda target, (sensor, value), count: "%s %s %s %.2f" % (target, sensor, value, count)
        output_for_target = lambda target: "\n".join([format_line(target, setting, count)
                                                      for setting, count in sorted(self.counts.loc[target].iteritems())])
        print "\n".join([output_for_target(target) for target in sorted(self.counts.index)])
