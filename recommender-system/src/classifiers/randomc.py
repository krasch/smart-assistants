import random

from base import BaseClassifier


class RandomClassifier(BaseClassifier):

      name="Random"

      def __init__(self,features,target_names):
          BaseClassifier.__init__(self,features,target_names)

      def fit(self, train_data,train_target):
          return self

      def predict(self, test_data):
          output=[]
          for d in test_data:
              possible= self.possible_targets(d)
              random.shuffle(possible)
              output.append(possible)
          return output
