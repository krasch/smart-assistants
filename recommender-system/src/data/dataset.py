from collections import defaultdict
from datetime import datetime,timedelta
import re
from ConfigParser import SafeConfigParser
import os.path
from ConfigParser import SafeConfigParser

from sklearn.feature_extraction import DictVectorizer
from sklearn.datasets.base import Bunch
import numpy
import pandas

"""
class UnknownDataset(Exception):
      def __init__(self,dataset):
          self.dataset=dataset

      def __str__(self):
          return "Unknown dataset: %s"%self.dataset

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

"""

def load_dataset(path_to_csv, path_to_config=None):

    def default_config():
        #default name is the name of the csv file without extension, e.g. "houseA.csv" -> "houseA"
        default_name = os.path.splitext(os.path.basename(path_to_config))[0]
        #per default do not exclude any sensors or services
        default_excluded_sensors = ""
        default_excluded_services = ""

        default_conf = SafeConfigParser()
        default_conf.add_section("basic")
        default_conf.set("basic", "name", default_name)
        default_conf.add_section("excludes")
        default_conf.set("excludes", "excluded_sensors", default_excluded_sensors)
        default_conf.set("excludes", "excluded_services", default_excluded_services)

        return  default_conf

    def read_config():

        #read config if exists or use defaults
        config = default_config()
        if not path_to_config is None:
            if not os.path.exists(path_to_config):
                raise ValueError("Could not find config file at %s " % os.path.abspath(path_to_config))
            config.read(path_to_config)

        #excluded sensors and services are given as a comma-separated string, convert to python list
        excluded_list = lambda exclude: [e.strip().replace("\"", "") for e in exclude.split(",") if len(e.strip()) > 0]

        return config.get("basic", "name"), \
               excluded_list(config.get("excludes", "excluded_sensors")), \
               excluded_list(config.get("excludes", "excluded_services"))

    name, excluded_sensors, excluded_services = read_config()
    events = pandas.read_csv(path_to_csv, parse_dates=["timestamp"])
    data = events_to_dataset(events, name, excluded_sensors, excluded_services)
    return data

def events_to_dataset(events, name, excluded_sensors, excluded_services):


    def extract_actions():

        #create dataframe with two columns, one for the actions and one for the timestamp of the actions occurrence
        action_name = lambda row: "%s=%s" % (row["sensor"], row["value"])
        actions_name_column = events.apply(action_name, axis=1)
        actions_with_timestamps = pandas.concat([actions_name_column, events["timestamp"]], axis=1)
        actions_with_timestamps.columns = ["action", "timestamp"]

        #todo explain the shift
        actions_with_timestamps = actions_with_timestamps.shift(-1)

        return actions_with_timestamps

    def data_for_sensor(sensor, actions):
        #get all the events for this sensor
        relevant_events = events[events["sensor"] == sensor]

        #forward fill the sensor data
        #e.g. if at T=0 sensor=value 1 and at T=10 sensor=value2, then fill all in-between T=(1-9) with value1
        relevant_events = relevant_events.reindex(events.index)
        relevant_events.fillna(method="pad", inplace=True)

        #calculate for each user action, how much time has passed since the sensor value changed
        time_passed = actions["timestamp"] - relevant_events["timestamp"]

        #create a dataframe that contains one row for every user action, one column with the current sensor value at the
        #the time of the action, one column with the time passed since the sensor value changed
        sensor_data = pandas.DataFrame({sensor: relevant_events["value"],
                                        "%s_timedelta" % sensor: time_passed})
        return sensor_data


    #remove events coming from any of the excluded sensors
    events_for_excluded = events["sensor"].isin(excluded_sensors)
    events = events[numpy.invert(events_for_excluded)]

    #get a dataframe with the user actions
    actions = extract_actions()

    #create dataset with one row per use action and two columns for each sensor:
    #column $sensor contains sensor values, column $sensor (timedelta) contains time passed since the sensor value
    #changed at the time of the action corresponding to the current row
    sensors = events["sensor"].unique()
    data = pandas.concat([data_for_sensor(sensor, actions) for sensor in sorted(sensors)], axis=1)

    #add action as a final column and set action timestamp as index
    data["action"] = actions["action"]
    data.index = actions["timestamp"]

    #drop data for actions that correspond to excluded services
    data_for_excluded = data["action"].isin(excluded_services)
    data = data[numpy.invert(data_for_excluded)]

    #for the last row, there is no next user actions -> remove that row
    data = data[0:-1]

    data.name = name
    return data


def dataset_to_scikit(data):

    #convert a nominal attribute to several binary features, one for each attribute value
    #e.g."door" [open/close] converts to "door=open" (can be 1 or 0) and "door=closed)" (can be 1 or 0)
    def attribute_to_binary(attribute_data):
        attribute_data = attribute_data.dropna()

        def binary_column_for_value(val):
            return attribute_data.apply(lambda v: 1 if v == val else 0)

        attribute_values = attribute_data.unique()
        binary_columns = [binary_column_for_value(value) for value in attribute_values]
        binary_columns = pandas.concat(binary_columns, axis=1)
        binary_columns.columns = ["%s=%s" % (attribute_data.name, value) for value in attribute_values]

        return binary_columns

    dataset_name = data.name

    #todo explain targets
    targets = data["action"]
    data = data.drop("action", axis=1)

    #todo why are integer indexes better?
    times = data.index
    data.reset_index(inplace=True)
    data = data.drop("timestamp", axis=1)

    #seperate columns that contain current sensor values from columns that contain timedelta information
    timedelta_columns = [col for col in data.columns if col.endswith("_timedelta")]
    value_columns = [col for col in data.columns if not col in timedelta_columns]

    #scikit does not support nominal attributes -> convert each attribute to several binary columns, one for each value
    binarized_data = pandas.concat([attribute_to_binary(data[attribute]) for attribute in value_columns], axis=1)

    #attach timedelta columns after binary columns
    binarized_data = pandas.concat([binarized_data, data[timedelta_columns]], axis=1)

    #scikit will give the classifiers only the data itself, without the column headers to make sense of the data
    #-> create an index of the columns that the classifier will be initalized with
    #todo improve when improving base classifier
    columns = {value: index for index, value in enumerate(binarized_data.columns)}
    index = []
    for attribute in value_columns:
        binary_column_names = [col for col in binarized_data.columns if col.startswith(attribute+"=")]
        timedelta_column_name = attribute+"_timedelta"
        for bin in binary_column_names:
            index.append((attribute, bin.replace(attribute+"="," "), columns[bin], columns[timedelta_column_name]))


    return Bunch(name=dataset_name,
                 data=binarized_data.values,
                 target=targets.values,
                 features=index,
                 times=times.values,
                 target_names=sorted(targets.unique()))


def write_dataset_as_arff(data, path_to_arff):

    def columns_as_arff_attributes():
        """
        The arff header contains information about all attributes (=columns) in the dataset. This method converts column
        metadata into the correct arff format.
        @return: A list of strings with the arff descriptions for all columns in the dataset.
        """
        value_column_to_attribute = lambda col: "@attribute %s {%s}" % (col, ",".join(data[col].dropna().unique()))
        timestamp_column_to_attribute = lambda col: "@attribute %s real" % col
        column_to_attribute = lambda col: timestamp_column_to_attribute(col) if "timedelta" in col \
                                          else value_column_to_attribute(col)

        return [column_to_attribute(column) for column in data.columns]

    def convert_timestamps(original):
        """
        Convert the numpy timedeltas (which are in nanoseconds) to seconds
        @param original: The original dataset with numpy timedeltas.
        @return: The dataset with timedeltas replaced
        """
        nanoseconds_to_seconds = lambda ts: str(int(ts.item()/(1000.0*1000.0*1000.0)))
        is_missing = lambda ts: str(ts) == "NaT"
        convert_timedelta = lambda ts: numpy.nan if is_missing(ts) else nanoseconds_to_seconds(ts)

        #perform the conversion
        converted = original
        timedelta_columns = filter(lambda col: isinstance(original[col][0], numpy.timedelta64), original.columns)
        converted[timedelta_columns] = converted[timedelta_columns].applymap(convert_timedelta)

        return converted

    def data_as_arff_strings():
        """
        Map each row in the dataset to a string in arff format.
        @return: A list of strings, one for each row in the dataset
        """

        #convert timestamps to format that arff can understand
        converted_data = convert_timestamps(data)

        #replace any N/A values with the arff symbol "?" for missing data
        converted_data = converted_data.fillna("?")

        #convert each row to one string, column values are separated by ","
        row_to_string = lambda row: ",".join(row.values)
        converted_data = converted_data.apply(row_to_string, axis=1)

        return converted_data.values

    f = open(path_to_arff, "w")

    #arff header contains the name of the dataset and one line for each attribute describing the attribute's type
    f.write("@relation %s\n" % data.name)
    f.write("\n".join(columns_as_arff_attributes()))

    #arff body has one line for each row in the dataset with the values of this row
    f.write("\n\n@data\n")
    f.write("\n".join(data_as_arff_strings()))
    f.close()



