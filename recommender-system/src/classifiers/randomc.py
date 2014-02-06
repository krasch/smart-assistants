import random

import pandas

from base import BaseClassifier


class RandomClassifier(BaseClassifier):
    """
    A classifier that gives random service recommendations, use as baseline in experiments.
    """

    name="Random"

    def __init__(self,features,target_names):
        BaseClassifier.__init__(self,features,target_names)

    def fit(self, train_data,train_target):
        return self

    def predict(self, test_data):

        #load test data into pandas dataframe and keep only the columns with current sensor settings
        test_data = pandas.DataFrame(test_data)
        test_data.columns = self.features
        test_data = test_data[self.settings_columns].values

        #to predict for instance: randomly order the possible targets
        def predict_for_instance(instance):
            currently_set = self.instance_settings(instance)
            possible_targets_mask = self.possible_targets_mask(currently_set)
            possible_targets = [target for target, is_possible
                                in zip(self.target_names, possible_targets_mask) if is_possible]
            random.shuffle(possible_targets)
            return possible_targets

        results = [predict_for_instance(test_data[i]) for i in range(len(test_data))]
        return results
