import random

import pandas

from base import BaseClassifier


class RandomClassifier(BaseClassifier):

    name="Random"

    def __init__(self,features,target_names):
        BaseClassifier.__init__(self,features,target_names)

    def fit(self, train_data,train_target):
        return self

    def predict(self, test_data):

        #load test data into pandas dataframe
        test_data = pandas.DataFrame(test_data)
        test_data.columns = self.features

        def predict_for_instance(instance):
            current_settings = self.current_settings(instance)
            possible_targets = self.possible_targets(current_settings)
            return random.shuffle(possible_targets)

        results = [predict_for_instance(row) for id, row in test_data.iterrows()]
        return results
