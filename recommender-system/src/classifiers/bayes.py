import pandas
import numpy

from base import BaseClassifier
from profilehooks import profile

def print_currently_set(currently_set):
    out_list=["%s=%s"%(attribute,value) for (attribute,value,timedelta) in sorted(currently_set)]
    print " ".join(out_list)

class NaiveBayesClassifier(BaseClassifier):

    name = "NaiveBayes"

    def __init__(self, features, target_names):
        BaseClassifier.__init__(self, features, target_names)

    def fit(self, train_data,train_target):

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
            return normalized_counts

        def calculate_counts_per_setting(data):
            """
            Count how often each target was seen in each settings.  Performs additive smoothing and normalizes
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
            return normalized_counts


        #load training data and targets into pandas dataframe
        train_data = pandas.DataFrame(train_data)
        train_data.columns = self.features
        train_data.index = train_target

        #keep only the value columns, since Naive Bayes does not use timedeltas
        value_columns, timedelta_columns = self.identify_column_types(train_data)
        train_data = train_data[value_columns]

        #calculate how often each target was seen and how often it was seen in specific settings
        self.priors = calculate_priors(train_data)
        self.counts = calculate_counts_per_setting(train_data)

        #self.print_counts_and_priors()

        return self

    #@profile
    def predict(self, test_data):

        #load test data into pandas dataframe
        test_data = pandas.DataFrame(test_data)
        test_data.columns = self.features

        #keep only the value columns, since Naive Bayes does not use timedeltas
        value_columns, timedelta_columns = self.identify_column_types(test_data)
        test_data = test_data[value_columns]

        #for one instance: calculate posteriors, normalize them and return sorted recommendations
        def predict_for_instance(instance):
            current_settings = self.current_settings(instance)
            possible_targets_mask = self.possible_targets_mask(current_settings)

            counts = self.counts[current_settings]
            posteriors = counts.prod(axis=1) * self.priors
            posteriors = (posteriors*possible_targets_mask).dropna()
            normalized_posteriors = posteriors.div(posteriors.sum())
            normalized_posteriors.sort(ascending=False)

            return normalized_posteriors.index.values

        results = [predict_for_instance(row) for id, row in test_data.iterrows()]

        return results


    def print_counts_and_priors(self):
        line_to_string = lambda target, count: "%s %.2f" % (target, count)
        print "\n".join([line_to_string(target, count) for target, count in self.priors.iteritems()])

        format_line = lambda target, (sensor, value), count: "%s %s %s %.2f" % (target, sensor, value, count)
        output_for_target = lambda target: "\n".join([format_line(target, setting, count)
                                                      for setting, count in sorted(self.counts.loc[target].iteritems())])
        print "\n".join([output_for_target(target) for target in sorted(self.counts.index)])
