from collections import defaultdict
from datetime import datetime,timedelta
import re

from sklearn.feature_extraction import DictVectorizer
from sklearn.datasets.base import Bunch
import numpy
import pandas

class UnknownDataset(Exception):
      def __init__(self,dataset):
          self.dataset=dataset

      def __str__(self):
          return "Unknown dataset: %s"%self.dataset

def action(sensor,value):
    return "%s=%s"%(sensor,value)

class Setting:

    def __init__(self,time,sensor,value):
        self.sensor=sensor
        self.value=value
        self.time=time

    def copy(self):
        return Setting(self.time,self.sensor,self.value)


    def __str__(self):
        if self.time is None: 
           return "%s=%s"%(self.sensor,self.value)
        else:
           return "%s=%s (%s)"%(self.sensor,self.value,self.time)


class Instance:

    def __init__(self,time):
        self.settings=dict()
        self.action=None
        self.time=time

    def copy(self,time):
        new=Instance(time)    
        for setting in self.settings.values():
            new.settings[setting.sensor]=setting.copy()
        return new

    def update_setting(self,time,sensor,value):
        self.settings[sensor]=Setting(time,sensor,value)

    def set_action(self,time,sensor,value):
        self.action=Setting(time,sensor,value)

    def setting_changed(self,sensor,value):
        if not sensor in self.settings:
           return True
        if self.settings[sensor].value==value:
           return False
        return True

    def value(self,sensor):
        if not sensor in self.settings:
           return None
        return self.settings[sensor].value

    def timedelta(self,sensor):
        if not sensor in self.settings:
           return None
        return self.action.time-self.settings[sensor].time

    def __str__(self):
        ordered = [str(self.settings[setting]) for setting in sorted(self.settings)]
        return ", ".join(ordered) +"->"+str(self.action)


class Dataset:
    
    def __init__(self,name):
        self.name=name
        self.sensors=defaultdict(list)
        self.actions=[]
        self.data=[]

    def _excluded(self,name,excluded):
        for ex in excluded:
            if re.match(ex,name):
               return True
        return False

    def read(self,datafile,exclude_sensors=[],exclude_actions=[]):
        events=pandas.read_csv(datafile,index_col=0,parse_dates=[0])

        instance=None
        for (time,sensor,value) in events.itertuples(index=True):
            if self._excluded(sensor,exclude_sensors): #or (not instance is None and not instance.setting_changed(sensor,value)):
               continue
            if not value in self.sensors[sensor]:
               self.sensors[sensor].append(value)
            
            if instance is None:
               instance=Instance(time)                    #this is the first event, there is no service
               instance.update_setting(time,sensor,value)  
               continue

            if not self._excluded(action(sensor,value),exclude_actions):
               instance.set_action(time,sensor,value)
               self.data.append(instance)
               instance=instance.copy(time)
               if not (sensor,value) in self.actions:
                  self.actions.append((sensor,value))

            instance.update_setting(time,sensor,value)

    def __str__(self):
         return "\n".join([str(d) for d in self.data])
 
def timedelta_seconds(timedelta):
    return timedelta.days*24*60*60+timedelta.seconds     

def dataset_to_arff(dataset,exclude_timedeltas=False,filename=None):


    def escape(string):
        return "%s"%string

    #arff header
    arff=["@relation "+dataset.name]
    for sensor in sorted(dataset.sensors):
        arff.append("@attribute %s {%s}"%(sensor,",".join(escape(value) for value in dataset.sensors[sensor])))
        if not exclude_timedeltas:
           arff.append("@attribute %s_timedelta real"%sensor)
    arff.append("@attribute action {%s}"%(",".join([escape(action(sensor,value)) for (sensor,value) in dataset.actions])))


    #arff data body
    arff.append("@data")
    for d in dataset.data:
        values=[]
        for sensor in sorted(dataset.sensors):
            #value attribute 
            value=d.value(sensor)
            if value is None:
               values.append("?")
            else:
               values.append(escape(value))
            #timedelta attribute
            if not exclude_timedeltas:
               timedelta=d.timedelta(sensor)
               if timedelta is None:
                  values.append("?")
               else:
                  values.append(str(timedelta_seconds(timedelta)))
        values.append(escape(action(d.action.sensor,d.action.value)))
        arff.append(",".join(values))

    arff="\n".join(arff)
    if not filename is None:
       f = open(filename,"w")
       f.write(arff)
       f.flush()
       f.close()
    return arff

     
def scikit_timeline_generator(features,times,data,targets):
    timedelta_indexes=list(set([timedelta_index for _,_,_,timedelta_index in features]))
    
    #get an instance array such that it has all values set as the current instance in the dataset, 
    #but timedeltas reset to match time of next instance
    def initialize_instance(current_instance,current_time,next_time):
        delta_in_seconds=(next_time-current_time).seconds+(next_time-current_time).days*24*60*60
        instance=numpy.array(current_instance, copy=True)
        for index in timedelta_indexes:
            if instance[index]>=0:
               instance[index]-=delta_in_seconds
        return instance

    #increment all timedeltas with given number of seconds
    def increment_instance(instance,increment_seconds):
       instance=numpy.array(instance, copy=True)
       for index in timedelta_indexes:
           if instance[index]>=0:
              instance[index]+=increment_seconds
       return instance
 
    #the actual generator  
    def generate(timedelta):
        time=times[0]
        i=0
        while i<len(times)-1:
            if times[i]<=time:
               instance=initialize_instance(data[i],times[i],times[i+1])
               i+=1
            time+=timedelta  
            instance=increment_instance(instance,timedelta.seconds)
            yield instance
            #if times[i]<=time:
            #   yield instance,targets[i+1]
            #else:
            #   yield instance,None
    
    return generate

def dataset_to_scikit(dataset):

    times=[]
    data_values=[]
    data_timedeltas=[]
    targets=[]
    target_names=[]
    for d in dataset.data:
        values=dict()
        timedeltas=[]
        for sensor in sorted(dataset.sensors):
            #value attributes
            value=d.value(sensor)
            if value is None:
               values[sensor]="?"
            else:
               values[sensor]=value
            #timedelta attribute
            timedelta=d.timedelta(sensor)
            if timedelta is None:
               timedeltas.append(-1)
            else:
               timedeltas.append(timedelta_seconds(timedelta))   
        times.append(d.time)    
        data_values.append(values)
        data_timedeltas.append(timedeltas)
        target=action(d.action.sensor,d.action.value)
        targets.append(target)
        if not target in target_names:
           target_names.append(target)
 

    #convert nominal attributes to numeric, using several binary features per attribue
    vec = DictVectorizer()
    data=vec.fit_transform(data_values).toarray()
    feature_names=vec.get_feature_names()
 
    #add timedelta attributes
    data=numpy.concatenate((data,numpy.array(data_timedeltas)),axis=1)
    for sensor in sorted(dataset.sensors):
        feature_names.append(sensor+"_timedelta")

    #create feature index
    features=[]
    for sensor in sorted(dataset.sensors):
        timedelta_index=feature_names.index(sensor+"_timedelta")
        for value in sorted(dataset.sensors[sensor]):
            feature_index=feature_names.index(sensor+"="+value)
            features.append((sensor,value,feature_index,timedelta_index))

              
    return Bunch(name=dataset.name,
                 data=data,
                 target=numpy.array(targets),
                 features=features,
                 times=times,
                 timeline=scikit_timeline_generator(features,times,data,targets),
                 target_names=sorted(target_names))


