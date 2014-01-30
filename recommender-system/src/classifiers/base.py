from sklearn.base import BaseEstimator
from profilehooks import  profile

import numpy
import pandas

import textwrap
from copy import copy

def apply_mask(values, mask):
    return [value for value, is_set in zip(values, mask) if is_set == 1]


class BaseClassifier(BaseEstimator):

    def __init__(self, features, target_names):
        self.features = features

        #identify which columns are settings columns and which columns are timedelta columns
        self.settings_columns = [col for col in features if isinstance(col, tuple)]
        self.timedelta_columns = [col for col in features if not col in self.settings_columns]

        self.target_names = sorted(target_names)
        self.targets_as_tuples = [tuple(target.split("=")) for target in self.target_names]

    def possible_targets_mask(self, current_settings):
        is_possible_target = lambda target: 0 if target in current_settings else 1
        mask = numpy.array([is_possible_target(target) for target in self.targets_as_tuples])
        return mask



