from collections import defaultdict
from itertools import compress
import textwrap

import pandas
import numpy

from profilehooks import profile

from base import BaseClassifier, apply_mask
from binners import StaticBinning,smooth
from DS import arithmetic_mean_combiner,simple_conjunctive_combiner
from postprocess import dynamic_cutoff
from caching import NoCache,CacheMiss



def print_combined_masses(masses,conflict,theta):
    out_list = ["%s %.4f" % (m, masses[m]) for m in sorted(masses)]
    out_list.append(", conflict=%.4f, theta=%.4f" % (conflict, theta))
    print textwrap.fill(" ".join(out_list), initial_indent="", subsequent_indent="    ", width=200)

class Source():

    def __init__(self, sensor, value, total_counts, temporal_counts):
        self.sensor = sensor
        self.value = value
        self.targets = total_counts.index

        #already unpack bin columns into numpy arrays, makes later calculations much faster
        self.total_counts = total_counts.values
        self.temporal_counts = [numpy.array(temporal_counts[col]) for col in temporal_counts.columns]

    def name(self):
        return self.sensor + "=" + self.value

    def max_temporal(self):
        return max([temporal.sum() for temporal in self.temporal_counts])

    def has_temporal_knowledge(self, bin):
        return (bin != 14)

    def counts(self, bin):
        if self.has_temporal_knowledge(bin):
           return self.temporal_counts[bin]
        else:
           return self.total_counts


    #@profile
    def calculate_masses(self, bin, possible_targets_mask, max_total, max_temporal):

        counts = self.counts(bin) * possible_targets_mask
        counts_sum = counts.sum()

        if self.has_temporal_knowledge(bin):
            weight = counts_sum/max_temporal
        else:
            weight = counts_sum/max_total * 0.0001

        multiplier = weight/counts_sum

        masses = counts * multiplier

        return masses


    def __str__(self):
        out = self.name() + "\n"
        for target_index, target in enumerate(self.targets):
            out += "   %s: %d" % (target, self.total_counts[target_index]) + "\n"
            out += str(["%.3f" % c for c in self.temporal_counts[target_index]]) + "\n"
        return out

    def __print_source_info__(self, counts, weight, masses):
        print "%s=%s (%.4f)" % (self.sensor, self.value, weight)
        masses_dict = {target: masses[t] for t, target in enumerate(self.targets)}
        counts_dict = {target: counts[t] for t, target in enumerate(self.targets)}
        out_list = ["%s:(%.4f,%.4f)" % (target, counts_dict[target], masses_dict[target])
                    for target in sorted(counts_dict)]
        print textwrap.fill(" ".join(out_list), initial_indent="    ", subsequent_indent="    ", width=200)


class TemporalEvidencesClassifier(BaseClassifier):

    name = "TemporalEvidences"

    def __init__(self, features, target_names, binning_method=StaticBinning(),
                 combiner=simple_conjunctive_combiner,postprocess=None,cache=NoCache()):
        BaseClassifier.__init__(self, features,target_names)
        self.binning_method=binning_method
        self.sources=dict()


    def digitize_timedeltas(self, values):
        return numpy.digitize(values.values, self.binning_method.bins)

    def fit(self, train_data, train_target):
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

        #load training data and targets into pandas dataframe
        train_data = pandas.DataFrame(train_data)
        train_data.columns = self.features
        train_data.index = train_target

        #discretize the timedeltas into the given bins
        train_data[self.timedelta_columns] = train_data[self.timedelta_columns].apply(self.digitize_timedeltas)

        #create a Source that knows how often each target has been observed when sensor=value
        def create_source_for_setting(sensor, value):

            def smoothed_temporal_counts(bins_for_target):
                #count how often current target was seen in each bin (bincount is much faster than pandas value_counts)
                counts = numpy.bincount(bins_for_target.values, minlength=len(self.binning_method.bins)+1)
                #the last item contains counts of timedeltas for which no bin could be found, remove that item
                counts = counts[0:-1]
                #counts[-1] = 0        #todo: bug in original program!
                #perform smoothing; sometimes smoothed contains negative values near 0, round those up to 0
                smoothed = smooth(counts).clip(0.0)
                return pandas.Series(smoothed, name=bins_for_target.name)


            #retrieve from train_data exactly those rows where the given sensor is set to the given value and select the
            #corresponding timedelta column for this sensor
            observations_for_setting = train_data[[(sensor, value), "%s_timedelta" % sensor]]
            observations_for_setting.columns = ["is_set", "bins"]
            observations_for_setting = observations_for_setting[observations_for_setting["is_set"] == 1]["bins"]

            #group the observations by the targets
            bins_grouped_by_target = observations_for_setting.groupby(observations_for_setting.index, sort=False)

            #count how often each target was seen overall and how often each target was seen in each bin
            total_counts = bins_grouped_by_target.sum()
            temporal_counts = bins_grouped_by_target.apply(smoothed_temporal_counts).unstack()

            #some targets may have never been observed in this setting, fill counts for these with 0.0
            total_counts = total_counts.reindex(self.target_names).fillna(0.0)
            temporal_counts = temporal_counts.reindex(self.target_names).fillna(0.0)

            return Source(sensor, value, total_counts, temporal_counts)

        #create one source for each setting (sensor=value) that occurs in the dataset
        self.sources = {(sensor, value): create_source_for_setting(sensor, value)
                        for sensor, value in self.settings_columns}

        #maximum number of total observations for any setting
        self.max_total = max(source.total_counts.sum() for source in self.sources.values())
        #maximum number of observations in any bin for any setting
        self.max_temporal = max(source.max_temporal() for source in self.sources.values())

        return self

    #@profile
    def predict(self, test_data, include_conflict_theta=False):

        #load test data into pandas dataframe
        test_data = pandas.DataFrame(test_data)
        test_data.columns = self.features

        #divide test data into current settings and current timedeltas
        test_data_settings = test_data[self.settings_columns].values
        test_data_timedeltas = test_data[self.timedelta_columns]

        #replace timedeltas with the respective bin index
        test_data_bins = test_data_timedeltas.apply(self.digitize_timedeltas, axis=1).values

        #make an index that allows fast lookup of index of the correct timedelta column for each sensor
        timedelta_column_for_sensor = {sensor: self.timedelta_columns.index("%s_timedelta" % sensor)
                                       for sensor, value in self.settings_columns}


        #test_timedeltas = test_timedeltas.replace({13:14}, axis=1)   #todo again the bug

        def predict_for_instance(instance, instance_bins):
            current_bin_for_sensor = lambda sensor: int(instance_bins[timedelta_column_for_sensor[sensor]])

g
            #find which sensor values are currently set and the timedelta bins for these sensors
            currently_set = apply_mask(self.settings_columns, instance)
            bins_for_current_settings = [current_bin_for_sensor(sensor) for sensor, value in currently_set]


            #calculate which targets (user actions) are currently possible (possible targets are represented by 1,
            #not currently possible targets are represented by 0)
            possible_targets_mask = self.possible_targets_mask(currently_set)

            #calculate the mass distributions for all current settings
            masses = [self.sources[setting].calculate_masses(bin, possible_targets_mask, self.max_total, self.max_temporal)
                      for setting, bin in zip(currently_set, bins_for_current_settings)]

            #combine the masses using Dempster's combination rule
            combined_masses, conflict, theta = simple_conjunctive_combiner(masses)

            #map resulting masses to the possible targets and sort
            recommendations = {target: target_mass for target, target_mass, target_is_possible
                               in zip(self.target_names, combined_masses, possible_targets_mask)
                               if target_is_possible}
            sorted_recommendations = sorted(recommendations, key=recommendations.get, reverse=True)

            #print_combined_masses(recommendations, conflict, theta)

            #print sorted_recommendations

            if include_conflict_theta:
                return sorted_recommendations, conflict, theta
            else:
                return sorted_recommendations

        #calculate recommendations for each instance in the dataset
        results = [predict_for_instance(test_data_settings[i], test_data_bins[i]) for i in range(len(test_data_bins))]

        return results



             
