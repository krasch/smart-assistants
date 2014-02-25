# -*- coding: UTF-8 -*-
"""
This module contains functions and classes that are used by all implemented classifiers
"""


from sklearn.base import BaseEstimator
import numpy


def apply_mask(values, mask):
    """
    Convenience method to filter out values from a list according to some binary masking array.
    @param values: The list that should be filtered.
    @param mask: The masking array, must have the same length as the `values` array.
    @return: A list that contains only the selected values.
    """
    return [value for value, is_set in zip(values, mask) if is_set == 1]


class BaseClassifier(BaseEstimator):
    """
    Class that collects important methods for interpreting the data that are common to all classifiers. All classifiers
    for our problem domain should be inherited from BaseClassifier.
    """
    def __init__(self, features, target_names):
        """
        Initialize the base classifier, all inheriting classes must call this base constructor.
        @param features: The features of this dataset. Datasets in the scikit-learn-format are a numpy matrix with one
        row for each instance in the dataset and one column for every feature. In our problem domain, the feature list
        contains all possible sensor settings (e.g. [(Fridge, Open), (Fridge, Closed), (TV, On), (TV, Off)), ...] and
        corresponding timedeltas for each sensor (e.g. ["Fridge_timedelta", "TV_timedelta", ...]
        @param target_names: A list of possible targets for the classifier. In our domain, targets correspond to
        possible user actions.
        @return:
        """
        self.features = features

        #identify which columns are settings columns and which columns are timedelta columns
        self.settings_columns = [col for col in features if isinstance(col, tuple)]
        self.timedelta_columns = [col for col in features if not col in self.settings_columns]

        self.target_names = sorted(target_names)
        self.targets_as_tuples = [tuple(target.split("=")) for target in self.target_names]

    def currently_set(self, instance):
        """
        Identify which of the possible settings are currently active. For example, if self.settings_columns =
        [(door, open), (door, closed), (window, open), (window, closed) and instance=[1, 0, NaN, NaN], then this
        method returns [(door, open)],
        @param instance: An array of binary values (plus NaN) that represent current settings. This array must have
        the same length as self.settings_columns and binary entries correspond to the entries in self.settings_columns.
        @return: A list of tuples, listing all currently active settings.
        """
        return apply_mask(self.settings_columns, instance)

    def possible_targets_mask(self, current_settings):
        """
        Calculate which of all available targets (i.e. user actions) are possible in the current situation. An action is
        only possible under some given sensor settings, e.g. a window can only be opened if it is currently closed.
        @param current_settings: A list of tuples, listing all currently active settings.
        @return: A list of tuples, listing all possible targets.
        """
        is_possible_target = lambda target: 0 if target in current_settings else 1
        mask = numpy.array([is_possible_target(target) for target in self.targets_as_tuples])
        return mask



