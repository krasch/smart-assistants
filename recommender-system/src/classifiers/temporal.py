from collections import defaultdict
import textwrap
import numpy

from profilehooks import profile

from base import BaseClassifier,normalize_dict
from binners import StaticBinning,smooth
from DS import arithmetic_mean_combiner,simple_conjunctive_combiner
from postprocess import dynamic_cutoff
from caching import NoCache,CacheMiss


 
def print_source_info((attribute,value,timedelta),counts,weight,masses,targets):
    print "%s=%s (%d) (%.4f)"%(attribute,value,timedelta,weight)
    masses_dict={target:masses[t] for t,target in enumerate(targets) if counts[t]>0}
    counts_dict={target:counts[t] for t,target in enumerate(targets) if counts[t]>0}
    #out_list=["%s:(%.2f,%.4f)"%(target,counts[target],masses[target]) for target in sorted(counts,key=counts.get,reverse=True)]
    out_list=["%s:(%.2f,%.4f)"%(target,counts_dict[target],masses_dict[target]) for target in sorted(counts_dict)]
    print textwrap.fill(" ".join(out_list),initial_indent="    ",subsequent_indent="    ",width=200)

def print_currently_set(currently_set):
    out_list=["%s=%s (%d)"%(attribute,value,timedelta) for (attribute,value,timedelta) in sorted(currently_set,key=lambda item : item[2])]
    print textwrap.fill(" ".join(out_list),initial_indent="",subsequent_indent="    ",width=200)

def print_combined_masses(masses,conflict,theta):
    out_list=["%s %.4f"%(m,masses[m]) for m in sorted(masses)]
    out_list.append(", conflict=%.4f,theta=%.4f"%(conflict,theta))
    print textwrap.fill(" ".join(out_list),initial_indent="",subsequent_indent="    ",width=200)

def print_possible_targets(mask,targets):
    out_list=[target for t,target in enumerate(targets) if mask[t]==1]
    print textwrap.fill(" ".join(out_list),initial_indent="",subsequent_indent="    ",width=200) 


class Source:

      def __init__(self,attribute,value,total_counts,temporal_counts):
          self.attribute=attribute
          self.value=value
          self.total_counts=total_counts
          self.temporal_counts=temporal_counts

      def print_source(self,targets):
          out=["%s=%s"%(self.attribute,self.value)]
          for t,target in enumerate(targets):
              if self.total_counts[t]>0:
                 out.append("   %s: %d"%(target,self.total_counts[t]))
                 temp=["%.3f"%self.temporal_counts[b][t] for b in range(len(self.temporal_counts))]
                 out.append(" %s "%str(temp))
          print "\n".join(out) 

      def counts(self,key,possible_mask):
          if key is None:
             counts_=self.total_counts
          else:
             counts_=self.temporal_counts[key]
          counts_=counts_*possible_mask
          return counts_

      #assign masses to all of the possible elements for this interval
      def calculate_mass_distribution(self,timedelta,key,possible_mask,max_temporal,max_total,targets): 
          observations=self.counts(key,possible_mask)
          observations_sum=observations.sum()
          if key is None:
             weight=float(observations_sum)/float(max_total)*0.0001   #has no temporal knowledge, additional discounting necessary
          else:
             weight=float(observations_sum)/float(max_temporal)       #has temporal knowledge
          factor=weight/observations_sum
          masses=observations*factor
          #print_source_info((self.attribute,self.value,timedelta),observations,weight,masses,targets)  
          return masses        

      def source_name(self):
          return self.attribute+"="+self.value


class TemporalEvidencesClassifier(BaseClassifier):

      name="TemporalEvidences"

      def __init__(self,features,target_names,binning_method=StaticBinning(),
                   combiner=simple_conjunctive_combiner,postprocess=None,cache=NoCache()):
          BaseClassifier.__init__(self,features,target_names)
          self.binning_method=binning_method
          self.sources=dict()
          self.combiner=combiner
          self.postprocess=postprocess
          self.cache=cache


      #perform training
      def fit(self, train_data,train_target):
          #perform binning and smoothing, convert into numpy array format and create sources
          #new format is [[bin0_target0,bin0_target1,...],[bin1_target0,bin1_target1,...]
          def make_source(attribute,value,temporal,total):
              num_bins=len(self.binning_method.bins) 
              temporal_indexed=[numpy.zeros(self.num_targets) for b in range(num_bins)]
              total_indexed=numpy.zeros(self.num_targets)
              for t,target in enumerate(self.target_names):                  
                  #has never been observed, keep zeros in the array
                  if not target in total:
                      continue
                  #perform binning and smoothing
                  binned=self.binning_method.perform_binning(temporal[target])
                  binned=smooth(numpy.array(binned)).clip(0.0000001)
                  #add to the index
                  for b,count in enumerate(binned):
                      temporal_indexed[b][t]=count
                  total_indexed[t]=total[(target)]
              return Source(attribute,value,total_indexed,temporal_indexed)

          #extract timedeltas and counts
          temporal=defaultdict(dict)
          total=defaultdict(dict)
          for i in range(len(train_data)):
              target=train_target[i]
              for (attribute,value,timedelta) in self.features_currently_set(train_data[i]):
                  total[(attribute,value)].setdefault(target,0)
                  total[(attribute,value)][target]+=1
                  temporal[(attribute,value)].setdefault(target,[])
                  temporal[(attribute,value)][target].append(timedelta)

	  #perform smoothing etc and create sources
          self.sources=dict() 
          self.max_total=0
          self.max_temporal=0
          for attribute,value in total:
              source_temporal=temporal[(attribute,value)]
              source_total=total[(attribute,value)]
              source=make_source(attribute,value,source_temporal,source_total)
              self.sources[(attribute,value)]=source
              self.max_total=max(self.max_total,source.total_counts.sum())
              for bin in source.temporal_counts:
                  self.max_temporal=max(self.max_temporal,bin.sum())
          self.cache.set_sources(self.sources,self.binning_method)   

          #for source in sorted(self.sources):
          #    self.sources[source].print_source(self.target_names)
          return self

      #@profile
      def predict(self, test_data,include_conflict_theta=False):
          results=[]
          for d in test_data:
              currently_set=self.features_currently_set(d)
              #print_currently_set(currently_set)
              try:
                 (masses,conflict,theta)=self.cache.get_cached_evidences(currently_set)
              except CacheMiss:
                 #calculate mass distributions for all sources
                 possible_mask=self.possible_targets_mask(d)
                 #print_possible_targets(possible_mask,self.target_names)
                 mass_distributions=[]

                 for (attribute,value,timedelta) in sorted(currently_set,key=lambda tup: tup[2]): 
                      if not (attribute,value) in self.sources:
                         continue
                      source=self.sources[(attribute,value)]
                      key=self.binning_method.key(timedelta)
                      masses=source.calculate_mass_distribution(timedelta,key,possible_mask,self.max_temporal,self.max_total,self.target_names)
                      mass_distributions.append(masses)  

                 #combine the sources
                 (masses,conflict,theta)=self.combiner(mass_distributions) 
                 masses={target:masses[i] for i,(is_possible,target) in enumerate(zip(possible_mask,self.target_names)) if is_possible} 
                 self.cache.update(currently_set,masses,conflict,theta)
              if not self.postprocess is None: 
                 masses=self.postprocess(masses,conflict,theta)
              #print_combined_masses(masses,conflict,theta)
              if include_conflict_theta:
                 results.append((self.sort_results(masses),conflict,theta))
              else:
                 results.append(self.sort_results(masses))
              #best=results[-1][0]
              #print best,masses[best]
          return results 
             
