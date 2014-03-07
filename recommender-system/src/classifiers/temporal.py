# -*- coding: UTF-8 -*-
"""
This module implements the proposed classifier, which identifies temporal relations between user actions to generate
service recommendations. The classifier makes use of Dempster-Shafer theory (see file DS.py) and temporal binning (see
binning.py).
"""

import textwrap

import pandas
import numpy

from profilehooks import profile

from base import BaseClassifier
from binning import smooth, initialize_bins
from DS import combine_dempsters_rule


u"""
As default, the classifier uses bins of width 10 seconds for timedelta ∆t ∈ [0, 60] and 30 seconds for ∆t ∈ (0, 300].
"""
default_bins = initialize_bins(0, 60, 10) + initialize_bins(60, 300, 30)

#numpy.seterr(all="warn")

class Source():
    """
    A source has information about some setting (sensor=value), e.g. how often user actions (=targets) where observed
    in this setting in general and how often they where observed in each bin.
    """

    #sources that have no temporal knowledge have insufficient data about the current situation and are discounted
    __no_temporal_knowledge_discount__ = 0.0001

    #if the bin index is -1, the source has no temporal knowledge about the current situation
    __has_temporal_knowledge__ = lambda self, bin: bin != -1

    def __init__(self, sensor, value, total_counts, temporal_counts):
        """
        Initialize the source with all necessary information about the setting represented  by this source.
        @param sensor: The sensor described by this source, is part of the setting represented by this source.
        @param value: Second part of the setting sensor=value that this source represents.
        @param total_counts: A pandas series of counts that state how often each target was observed in this setting;
        the series is indexed by the names of the targets.
        @param temporal_counts: A pandas dataframe with one column for every temporal bin. Each column contains counts
        that state how often each target was observed in this setting in the respective bin. The dataframe is indexed
        by the names of the targets.
        """
        self.sensor = sensor
        self.value = value
        self.targets = total_counts.index

        #already unpack bin columns into numpy arrays, makes later calculations much faster
        self.total_counts = total_counts.values
        self.temporal_counts = [numpy.array(temporal_counts[col]) for col in temporal_counts.columns]

    def name(self):
        """
        Obtain a printable name for the source, this name equivalent to the setting this source represents.
        @return: The name of the source.
        """
        return self.sensor + "=" + self.value

    def max_temporal(self):
        """
        Calculate the maximum number of observations in any of the temporal intervals.
        @return: The calculated maximum.
        """
        return max([temporal.sum() for temporal in self.temporal_counts])

    def counts(self, bin):
        """
        Lookup how often the targets where observed in the current situation.
        @param bin: The temporal bin for the current timedelta for the sensor represented by this source (reflects
        how long the sensor setting has not changed).
        @return: A numpy array with the same length as self.targets, each item in the result array represents how often
        the corresponding target has been observed in the current situation.
        """
        if self.__has_temporal_knowledge__(bin):
            return numpy.copy(self.temporal_counts[bin])
        else:
            return numpy.copy(self.total_counts)

    #@profile
    def calculate_masses(self, bin, possible_targets_mask, max_total, max_temporal):
        """
        Calculate masses that reflect how probable each target is in the current situation, according to this source.
        @param bin: The temporal bin for the current timedelta for the sensor represented by this source (reflects
        how long the sensor setting has not changed).
        @param possible_targets_mask: A numpy array with the same length as self.targets; if an item in the array is 1.0
        it means that the corresponding target is possible in the current situation, if an item is 0.0, then the
        corresponding target is not possible.
        @param max_total: Maximum number of total observations by source.
        @param max_temporal: Maximum number of observations by any source in any bin.
        @return: A numpy array with the same length as self.targets; each item in the result array represents the masses
        that the source attributes to the corresponding target.
        """

        #for all targets look up how many observations this source has in the current bin
        counts = self.counts(bin)

        #keep only observations for currently possible targets
        counts *= possible_targets_mask

        #calculate the current weight of this source
        counts_sum = counts.sum()
        if self.__has_temporal_knowledge__(bin):
            weight = counts_sum/max_temporal
        else:
            weight = counts_sum/max_total * self.__no_temporal_knowledge_discount__

        #calculate the mass distribution for the possible targets
        if counts_sum == 0:
            masses = counts
        else:
            masses = counts * (weight/counts_sum)

        return masses

    def __str__(self):
        """
        For debugging, produces debug output of the total and temporal observations for all targets by this source.
        """
        out = self.name() + "\n"
        for target_index, target in enumerate(self.targets):
            out += "   %s: %d" % (target, self.total_counts[target_index]) + "\n"
            out += str(["%.3f" % self.temporal_counts[bin][target_index]
                        for bin in range(len(self.temporal_counts))]) + "\n"
        return out

    def __print_source_info__(self, counts, weight, masses):
        """
        For debugging, produces output that summarizes what a source knows about a given situation.
        """
        print "%s=%s (%f)" % (self.sensor, self.value, weight)
        masses_dict = {target: masses[t] for t, target in enumerate(self.targets)}
        counts_dict = {target: counts[t] for t, target in enumerate(self.targets)}
        out_list = ["%s:(%.6f,%.6f)" % (target, counts_dict[target], masses_dict[target])
                    for target in sorted(counts_dict)]
        print textwrap.fill(" ".join(out_list), initial_indent="    ", subsequent_indent="    ", width=200)


class TemporalEvidencesClassifier(BaseClassifier):

    """
    Implements the proposed classifier that uses temporal relationships between user actions to recommend services.
    """
    name = "TemporalEvidences"

    def __init__(self, features, target_names, bins=default_bins, postprocess=None):
        """
        Initialize the classifier.
        @param features: see `BaseClassifier.__init__()`.
        @param target_names: see `BaseClassifier.__init__()`.
        @param bins: A list of interval borders as generated by `initialize_bins`.
        @param postprocess: A function that can be called to postprocess the generated recommendations in some manner.
        At the moment, static cutoff and dynamic cutoff are defined as postprocessing methods.
        @return:
        """
        BaseClassifier.__init__(self, features, target_names)
        self.bins = bins
        self.postprocess = postprocess

        #make an index that allows fast lookup of index of the correct timedelta column for each sensor
        self.timedelta_column_for_sensor = {sensor: self.timedelta_columns.index("%s_timedelta" % sensor)
                                            for sensor, value in self.settings_columns}

    def digitize_timedeltas(self, values):
        """
        Map each timedelta between two user actions/sensor changes to the respective bin index for this timedelta.
        @param values: A numpy array of timedeltas.
        @return: A numpy array of bin indexes.
        """
        digitized = numpy.digitize(values.values, self.bins)
        #the last bin contains all values that can not be placed in a regular bin
        digitized[digitized == len(self.bins)] = -1
        return digitized

    def fit(self, train_data, train_target):
        """
        Train the classifier.
        @param train_data: A matrix with len(self.features) columns and one row for each instance in the dataset. Each
        row describes a user situation with current sensor settings and information on how long these settings have
        not changed. See `dataset.dataset_to_sklearn` for more details.
        @param train_target: An array of targets (length of the array corresponds to the number of rows in train_data).
        Each target represents the action that user performed in the situation described by the corresponding row
        in train_data. See `dataset.dataset_to_sklearn` for more details.
        @return: self-reference for this classifier
        """

        #load training data and targets into pandas dataframe
        train_data = pandas.DataFrame(train_data)
        train_data.columns = self.features
        train_data.index = train_target

        #discretize the timedeltas into the given bins
        train_data[self.timedelta_columns] = train_data[self.timedelta_columns].apply(self.digitize_timedeltas)

        #create one source for each setting (sensor=value) that occurs in the dataset
        self.sources = {(sensor, value): self.__create_source_for_setting__(sensor, value, train_data)
                        for sensor, value in self.settings_columns}

        #maximum number of total observations for any setting
        self.max_total = float(max(source.total_counts.sum() for source in self.sources.values()))
        #maximum number of observations in any bin for any setting
        self.max_temporal = float(max(source.max_temporal() for source in self.sources.values()))

        return self

    def __create_source_for_setting__(self, sensor, value, train_data):
        """
        Create a Source that knows how often each target has been observed when sensor=value
        """

        def smooth_temporal_counts(bins_for_target):
            #if the bin index is -1, the timedelta could not be placed in a regular bin -> remove these observations
            cleaned_bins_for_target = bins_for_target.values
            cleaned_bins_for_target = cleaned_bins_for_target[cleaned_bins_for_target >= 0]
            #count how often current target was seen in each bin (bincount is much faster than pandas value_counts)
            counts = numpy.bincount(cleaned_bins_for_target, minlength=len(self.bins))
            #counts[-1] = 0        #if this line is uncommented it reproduces a bug in the original program
            #perform smoothing; sometimes smoothed contains negative values near 0, round those up to 0
            smoothed = smooth(counts).clip(0.0)
            return pandas.Series(smoothed, name=bins_for_target.name)

        def calculate_total_counts(bins_grouped_by_target):
            if len(bins_grouped_by_target) > 0:
               total_counts = bins_grouped_by_target.count()
               #some targets may have never been observed in this setting, fill counts for these with 0.0
               total_counts = total_counts.reindex(self.target_names).fillna(0.0)
            else:
               #have no observations at all for this setting
               total_counts = pandas.Series(0.0, index=self.target_names)
            return total_counts

        def calculate_temporal_counts(bins_grouped_by_target):
            if len(bins_grouped_by_target) > 0:
               temporal_counts = bins_grouped_by_target.apply(smooth_temporal_counts).unstack()
               #some targets may have never been observed in this setting, fill counts for these with 0.0
               temporal_counts = temporal_counts.reindex(self.target_names).fillna(0.0)
            else:
               #have no observations at all for this setting
               temporal_counts = pandas.DataFrame(0.0, index=self.target_names, columns=range(0, len(self.bins)))
            return temporal_counts


        #retrieve from train_data exactly those rows where the given sensor is set to the given value and select the
        #corresponding timedelta column for this sensor
        observations_for_setting = train_data[[(sensor, value), "%s_timedelta" % sensor]]
        observations_for_setting.columns = ["is_set", "bins"]
        observations_for_setting = observations_for_setting[observations_for_setting["is_set"] == 1]["bins"]

        #group the observations by the targets
        bins_grouped_by_target = observations_for_setting.groupby(observations_for_setting.index, sort=False)

        #count how often each target was seen overall and how often each target was seen in each bin
        total_counts = calculate_total_counts(bins_grouped_by_target)
        temporal_counts = calculate_temporal_counts(bins_grouped_by_target)

        return Source(sensor, value, total_counts, temporal_counts)

    #@profile
    def predict(self, test_data, include_conflict_theta=False):
        """
        Calculate service recommendations for each instance in the test dataset.
        @param test_data: A matrix with len(self.features) columns and one row for each instance in the dataset. Each
        row describes a user situation with current sensor settings and information on how long these settings have
        not changed. More details on the setup of each row can be found in `self.__predict_for_instance__`.
        @param include_conflict_theta: If this parameter is false, the function returns only the service recommendations.
        If this parameter is true, it returns also information on recommendation conflict and uncertainty (theta).
        @return: Service recommendations for each instance in the test dataset, optionally also returns recommendation
        conflict and uncertainty for each instance.
        """

        #load test data into pandas dataframe
        test_data = pandas.DataFrame(test_data)
        test_data.columns = self.features

        #divide test data into current settings and current timedeltas
        test_data_settings = test_data[self.settings_columns].values
        test_data_timedeltas = test_data[self.timedelta_columns]

        #replace timedeltas with the respective bin index
        test_data_bins = test_data_timedeltas.apply(self.digitize_timedeltas, axis=1)
        #test_data_bins = test_data_bins.replace({len(self.bins): -1}, axis=1)  #if this line is uncommented it reproduces a bug in the original program
        test_data_bins = test_data_bins.values

        #calculate recommendations for each instance in the dataset
        results = [self.__predict_for_instance__(test_data_settings[i], test_data_bins[i], include_conflict_theta)
                   for i in range(len(test_data_bins))]

        return results

    def __predict_for_instance__(self, instance, instance_bin, include_conflict_theta):
        """
        @param instance: A numpy array with possible values [0, 1 and NaN] with the same length as self.settings_columns.
        Each entry in the array describes whether the corresponding setting is currently active (1), is not active (0)
        or the status of the sensor is not known (NaN).
        @param instance_bin: A numpy array of bin indexes with the same length as self.timedelta_columns. Each entry
        in the array describes how long the corresponding sensor has had the current value.
        @param include_conflict_theta: If this parameter is false, the function returns only the service recommendations.
        If this parameter is true, it returns also information on recommendation conflict and uncertainty (theta).
        @return: Service recommendations for this instance, optionally including conflict and uncertainty.
        """

        #find which sensor values are currently set and the timedelta bins for these sensors
        currently_set = self.currently_set(instance)
        current_bin_for_sensor = lambda sensor: int(instance_bin[self.timedelta_column_for_sensor[sensor]])
        bins_for_current_settings = [current_bin_for_sensor(sensor) for sensor, value in currently_set]

        #calculate which targets (user actions) are currently possible (possible targets are represented by 1,
        #not currently possible targets are represented by 0)
        possible_targets_mask = self.possible_targets_mask(currently_set)

        #calculate the mass distributions for all current settings
        masses = [self.sources[setting].calculate_masses(bin, possible_targets_mask, self.max_total, self.max_temporal)
                  for setting, bin in zip(currently_set, bins_for_current_settings)]

        #combine the masses using Dempster's combination rule
        combined_masses, conflict, theta = combine_dempsters_rule(masses)

        #map resulting masses to the possible targets, apply postprocessing and sort recommendations
        recommendations = {target: target_mass for target, target_mass, target_is_possible
                           in zip(self.target_names, combined_masses, possible_targets_mask)
                           if target_is_possible}
        if not self.postprocess is None:
            recommendations = self.postprocess(recommendations, conflict, theta)
        sorted_recommendations = sorted(recommendations, key=recommendations.get, reverse=True)

        if include_conflict_theta:
            return sorted_recommendations, conflict, theta
        else:
            return sorted_recommendations

             
def configure_static_cutoff(cutoff):
    """
    Configure a function that shortens the recommendations list to contain only the best `cutoff` recommendations
    @param cutoff: The number of recommendations to return.
    @return:  Function that can be called to actually perform the static cutoff for a list of generated recommendations.
    """
    def perform_static_cutoff(recommendations, conflict=None, theta=None):
        ordered = sorted(recommendations.items(), key=lambda (element, mass): mass, reverse=True)
        return {key: value for (key, value) in ordered[0:cutoff]}
    return perform_static_cutoff


def configure_dynamic_cutoff(max_conflict, max_theta, cutoff):
    """
    Configure a function that dynamically shortens the recommendations list if requirements for conflict and theta
    are fulfilled.
    @param max_conflict: If conflict is higher than max_conflict, do not perform the dynamic cutoff.
    @param max_theta: If theta is higher than max_cutoff, do not perform the dynamic cutoff.
    @param cutoff: If requirements for conflict and theta are fulfilled, return only the best `cutoff` elements.
    @return: Function that can be called to actually perform the dynamic cutoff for a list of generated recommendations.
    """
    static_cutoff = configure_static_cutoff(cutoff)

    def perform_dynamic_cutoff(recommendations, conflict, theta):
        if conflict < max_conflict and theta < max_theta:
            return static_cutoff(recommendations)
        else:
            return recommendations
    return perform_dynamic_cutoff
