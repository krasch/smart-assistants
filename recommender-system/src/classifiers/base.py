from sklearn.base import BaseEstimator
import numpy
from profilehooks import profile

import textwrap
from copy import copy

def normalize_dict(values):
    vsum=float(sum(values.values()))
    normalized=dict()
    for key in values:
        normalized[key]=float(values[key]+1)/(vsum+len(values))
    return normalized 


class BaseClassifier(BaseEstimator):

      def __init__(self,features,target_names):
          self.features=features
          self.target_names=sorted(target_names)
          self.num_targets=len(self.target_names)
          #maps the index of a target in the target_name vector to the corresponding index in the feature vector 
          self.target_to_feature={}
          for (attribute,value,index,timedelta_index) in self.features:
              target=self.get_target(attribute,value)
              if target in self.target_names:
                 self.target_to_feature[self.target_names.index(target)]=index


      def instance_string(self,instance):
          result=self.features_currently_set(instance)
          result=["%s=%s(%f)"%r for r in result]
          return ",".join(result)


      def features_currently_set(self,instance):
          result=[]
          for (attribute,value,index,timedelta_index) in self.features:
              if int(instance[index])==1:
                 result.append((attribute,value,instance[timedelta_index]))    
          return result


      #@profile
      def possible_targets_mask(self,instance):
          mask=numpy.zeros(self.num_targets)
          for t,target in enumerate(self.target_names):
              feature=self.target_to_feature[t]
              if int(instance[feature])==0:
                  mask[t]=1
          return mask

      def possible_targets(self,instance):
          mask=self.possible_targets_mask(instance)
          targets=[t for m,t in zip(mask,self.target_names) if m==1]
          return targets
          

      def get_target(self,attribute,value):
          return attribute+"="+value

      def sort_results(self,results):
          return sorted(results,key=results.get,reverse=True)



