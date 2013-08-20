from collections import defaultdict
import textwrap

from base import BaseClassifier,normalize_dict

def print_currently_set(currently_set):
    out_list=["%s=%s"%(attribute,value) for (attribute,value,timedelta) in sorted(currently_set)]
    print " ".join(out_list)

class NaiveBayesClassifier(BaseClassifier):

      name="NaiveBayes"

      def __init__(self,features,target_names):
          BaseClassifier.__init__(self,features,target_names)

      def fit(self, train_data,train_target):

          self.priors=dict()   #counts how often each target was seen
          self.counts=dict()   #counts how often each target was seen under specific circumstances

          #prepare priors and counts, necessary because we want all combinations in counts, even if not appearing in training data
          for target in self.target_names:
              self.priors[target]=0
              self.counts[target]=defaultdict(dict)
              for (attribute,value,index,timedelta_index) in self.features:
                  self.counts[target][attribute][value]=0             

          #do the actual counting
          for i in range(len(train_data)):
              target=train_target[i]
              self.priors[target]+=1
              for (attribute,value,timedelta) in self.features_currently_set(train_data[i]):
                  self.counts[target][attribute][value]+=1

          #perform normalization
          self.priors=normalize_dict(self.priors)
          for target in self.counts:
              for attribute in self.counts[target]: 
                 self.counts[target][attribute]=normalize_dict(self.counts[target][attribute])

          #self.print_counts_and_priors()
 
          return self 

      def predict(self, test_data):
          results=[]
          for d in test_data:
              currently_set=self.features_currently_set(d)
              #calculate posteriors
              posteriors=dict()
              for target in self.possible_targets(d):
                  posteriors[target]=1.0
                  for (attribute,value,timedelta) in currently_set:
                      posteriors[target]*=self.counts[target][attribute][value]    
                  posteriors[target]*=self.priors[target]

              #normalize the posteriors
              posteriorsum=sum(posteriors.values())
              for change in posteriors:
                  posteriors[change]/=posteriorsum
              
              results.append(self.sort_results(posteriors))
          return results

      def print_counts_and_priors(self):
          for target in sorted(self.priors.keys()):
             print "%s %.2f"%(target,self.priors[target])
          for target in sorted(self.counts.keys()):
            for attribute in sorted(self.counts[target].keys()):
                  for value in sorted(self.counts[target][attribute]):
                      print "%s %s %s %.2f"%(target,attribute,value,self.counts[target][attribute][value])
