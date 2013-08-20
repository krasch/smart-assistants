from datetime import datetime,timedelta
import random
import copy
from collections import defaultdict
import matplotlib.pyplot as plt

import numpy

from data.dataset import Dataset,Instance,dataset_to_scikit
from classifiers.temporal import TemporalEvidencesClassifier,Source
from experiment import avg_std_CI

class SyntheticDataset:
      
     def __init__(self,dimensions,size_nominal):
         self.dimensions=dimensions
         self.size_nominal=size_nominal
         self.__generate_sensors()

     def __generate_sensors(self):
         self.sensors={}
         self.actions=[]
         for i in range(self.dimensions):
             self.sensors["dim%s"%i]=["v%s"%j for j in range(self.size_nominal)]
             self.actions+=[("dim%s"%i,"v%s"%j) for j in range(self.size_nominal)]

     def generate_instances(self,num_instances):
         def generate_instance():
             itime=datetime.now()
             instance=Instance(itime)
             eligible_actions=copy.copy(self.actions)
             for s in self.sensors:
                 t=timedelta(seconds=random.randint(1,400))
                 v=random.choice(self.sensors[s])
                 instance.update_setting(itime-t,s,v)
                 eligible_actions.remove((s,v))
             action=random.choice(eligible_actions)
             instance.set_action(itime,action[0],action[1])
             return instance

         instances=[]
         for i in range(0,num_instances):
             instances.append(generate_instance())
         return instances


     def make_scikit_dataset(self,num_instances):
         instances=self.generate_instances(num_instances)
         dataset=Dataset("synthetic")
         dataset.sensors=self.sensors
         dataset.actions=self.actions
         dataset.data=instances
         return dataset_to_scikit(dataset)

     #skip the lenghty training process, just generate random sources
     def get_trained_classifier(self):
         #classifier needs feature vector with info on features and their indexes in the data
         def get_feature_vector():
             #generate self.size_nominal instances, so that each value 
             # for each sensor is used at least once
             def generate_full_range_instances():
                  instances=[]
                  for i in range(self.size_nominal):
                      instances.append(Instance(datetime.now()))
                      for s,values in self.sensors.items():
                          instances[-1].update_setting(datetime.now(),s,values[i])
                      target=random.choice(self.sensors.keys())
                      instances[-1].set_action(instances[-1].time,target,self.sensors[target][i])
                  return instances

             dataset=Dataset("synthetic")
             dataset.sensors=self.sensors
             dataset.actions=self.actions
             dataset.data=generate_full_range_instances()
             return dataset_to_scikit(dataset).features

         #generate random observations
         def generate_counts(cls): 
             sources={}
             max_total=0
             max_temporal=0
             num_actions=len(self.actions)
             num_bins=cls.binning_method.bins
             for sensor,values in self.sensors.items():
                 for v in values:
                     total=numpy.random.randint(10000,10030,num_actions)
                     temporal=[numpy.random.randint(1,30,num_actions) for b in range(len(num_bins))]
                     source=Source(sensor,value,total,temporal)     
                     sources[(sensor,v)]=source
                     max_total=max(max_total,source.total_counts.sum())
                     for bin in source.temporal_counts:
                         max_temporal=max(max_temporal,bin.sum())
             cls.sources=sources  
             cls.max_total=max_total
             cls.max_temporal=max_temporal
             cls.cache.set_sources(cls.sources,cls.binning_method)

         def vectorizer(features):
             def vectorize_instance(instance):
                 vector=numpy.zeros(features[-1][-1]+1)
                 for (attribute,value,index,timedelta_index) in features:
                     if instance.settings[attribute].value==value:
                        vector[index]=1 
                        vector[timedelta_index]=(instance.time-instance.settings[attribute].time).seconds
                 return vector
             def vectorize_instances(instances):
                 return [vectorize_instance(i) for i in instances]
             return vectorize_instances

         features=get_feature_vector()
         cls=TemporalEvidencesClassifier(features,["%s=%s"%(attribute,value) for (attribute,value) in self.actions])
         generate_counts(cls)
         return cls,vectorizer(features)
  
   
def plot_results(xs,results,filename):
    def text_format(object_):
        return
    legend=[]
    max_=0.0
    for (name,ys) in results:
        plt.plot(xs,ys,markersize=10,lw=2)  
        legend.append(name)
        max_=max(max_,max(ys))
    if max(xs)>1.0:
       plt.xlim(min(xs),max(xs))
    else:
       plt.xlim(0.0,1.0)
    text_format(plt.ylabel("runtime [ms]"))
    text_format(plt.xlabel('Number of dimensions'))
    text_format(plt.yticks())
    text_format(plt.xticks())
    plt.ylim(0,max_)
    plt.legend(legend,loc="upper left")
    plt.savefig(filename)
    plt.close()

def print_results(results,dimensions,sizes):
    #print "\t\t".join([".."]+["%d"%size for size in sizes])
    #for dims in dimensions:
    #    print "\t".join(["%d"%dims]+["%.2f,%.4f"%(results[dims][size][0],results[dims][size][2]) for size in sizes])    
    for dims in dimensions:
        for size in sizes:
            out=[str(dims),str(size*dims),"$%.2f\pm%.2f$"%(results[dims][size][0],results[dims][size][2])]
            print "  &  ".join(out)+"\\\\"

def predict_time_experiment():
    dimensions=[50,100,250,500,1000]
    sizes=[2,5]
    replications=25
    all_results=defaultdict(dict)
    for dims in dimensions:
        for size in sizes:
            dataset=SyntheticDataset(dimensions=dims,size_nominal=size)
            cls,vectorizer=dataset.get_trained_classifier()
            results=[]
            for i in range(replications):
                test_data=vectorizer(dataset.generate_instances(1))
                time=datetime.now()
                cls.predict(test_data)
                results.append((datetime.now()-time).microseconds/1000.0)
            all_results[dims][size]=avg_std_CI(results)
        print_results(all_results,[dims],sizes)
    for_plot=[]
    for size in sizes:
        for_plot.append((size,[all_results[dims][size][0] for dims in dimensions]))
    plot_results(dimensions,for_plot,"../plots/synthetic_predict.pdf")

predict_time_experiment()

#dataset=SyntheticDataset(dimensions=10,size_nominal=3)
#dataset=dataset.make_scikit_dataset(1000)
#cls = TemporalEvidencesClassifier(dataset.features,dataset.target_names)
#print cls
#cls = cls.fit(dataset.data, dataset.target)






