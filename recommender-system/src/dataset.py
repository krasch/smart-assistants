# -*- coding: UTF-8 -*-
"""
This module contains functions for reading datasets and converting them to the scikit-learn and arff (weka) formats.
"""

import os.path
from ConfigParser import SafeConfigParser

from sklearn.datasets.base import Bunch
import numpy
import pandas


def load_dataset(path_to_csv, path_to_config=None):
    """
    This function reads an event-list dataset and returns a dataset that lists for all event timestamps the current
    settings of all available sensors. Please see `events_to_dataset` for more information on the resulting dataset.
    @param path_to_csv: The csv file that contains the dataset. The data must be formatted in three columns:
    "timestamp", "sensor", "value". The timestamp must be in a format that is readable by pandas.
    @param path_to_config: Path where an optional config file can be found. Please look at the file "houseA.config" for
    how to structure this file.
    @return: The resulting dataset.
    """

    def default_config():
        #default name is the name of the csv file without extension, e.g. "houseA.csv" -> "houseA"
        default_name = os.path.splitext(os.path.basename(path_to_csv))[0]
        #per default do not exclude any sensors or services
        default_excluded_sensors = ""
        default_excluded_actions = ""

        default_conf = SafeConfigParser()
        default_conf.add_section("basic")
        default_conf.set("basic", "name", default_name)
        default_conf.add_section("excludes")
        default_conf.set("excludes", "excluded_sensors", default_excluded_sensors)
        default_conf.set("excludes", "excluded_actions", default_excluded_actions)

        return default_conf

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
               excluded_list(config.get("excludes", "excluded_actions"))

    name, excluded_sensors, excluded_services = read_config()
    events = pandas.read_csv(path_to_csv, parse_dates=["timestamp"])
    data = events_to_dataset(events, name, excluded_sensors, excluded_services)
    return data


def load_dataset_as_sklearn(path_to_csv, path_to_config=None):
    """
    This function reads an event-list dataset and returns a dataset according to the scikit-learn dataset format. This
    dataset be used to train and test the recommendation classifiers.
    @param path_to_csv: The csv file that contains the dataset. The data must be formatted in three columns:
    "timestamp", "sensor", "value". The timestamp must be in a format that is readable by pandas.
    @param path_to_config: Path where an optional config file can be found. Please look at the file "houseA.config" for
    how to structure this file.
    @return: The resulting dataset, see `dataset_to_sklearn`.
    """
    data = load_dataset(path_to_csv, path_to_config)
    return dataset_to_sklearn(data)


def events_to_dataset(events, name, excluded_sensors, excluded_actions):
    """
    Convert an event-list dataset and return a dataset that lists for all event timestamps the next user action, the
    current settings of all available sensors at the time of this next user action and for how long the sensors had their
    settings at the time of the user action.

    Example input event-list dataset:
                   timestamp   sensor  value
    ----------------------------------------
     0   2012-05-01 00:00:00  sensor1     on
     1   2012-05-02 00:00:04  sensor3    off
     2   2012-05-03 00:00:07  sensor1    off
     3   2012-05-01 00:00:09  sensor2     on
     4   2012-05-01 00:00:12  sensor3     on

    Resulting dataset for this event-list:
       sensor1  sensor1_timedelta sensor2  sensor2_timedelta sensor3  sensor3_timedelta       action    action_timestamp
    --------------------------------------------------------------------------------------------------------------------
     0      on           00:00:04       -                  -       -                 -  sensor3=off  2012-05-01 00:00:04
     1      on           00:00:07       -                  -     off          00:00:03  sensor1=off  2012-05-01 00:00:07
     2     off           00:00:02       -                  -     off          00:00:05   sensor2=on  2012-05-01 00:00:09
     3     off           00:00:05      on           00:00:03     off          00:00:08   sensor3=on  2012-05-01 00:00:12


    Each sensor column contains the current value of the sensor at the time of the action. The initial value of the
    sensor is not known, therefore the sensor value is set to missing until the first event for this sensor.

    Each sensor_timedelta columns lists how long the corresponding sensor value has not changed at the time of the
    user actions.

    The action column contains the user actions, the action_timestamp column contains the time where the action occurred.
    For simplicity, actions are named as "sensor=value". For example if the user turns off the TV, the internal status
    sensor of the TV will read "off" and this action is called "TV=off". Some sensor settings can not be mapped to
    actions, e.g. "ToiletFlush=Off" happens automatically without any user action. The excluded_actions parameter is the
    list of all such non-valid actions, all rows that contain these "actions" are removed from the resulting dataset.

    For the first user action (in this example "sensor1=on"), we do not yet have any sensor settings. For this reason
    this action is omitted from the dataset.

    @param events: The event-list dataset.
    @param name: The name of the dataset.
    @param excluded_sensors: A list of sensors that should not be included in the resulting dataset
    @param excluded_actions: A list of non-valid actions that should not be included in the resulting dataset.
    @return: The resulting dataset.
    """

    def extract_actions():
        #create dataframe with two columns, one for the actions and one for the timestamp of the actions occurrence
        action_name = lambda row: "%s=%s" % (row["sensor"], row["value"])
        actions_name_column = events.apply(action_name, axis=1)
        actions_with_timestamps = pandas.concat([actions_name_column, events["timestamp"]], axis=1)
        actions_with_timestamps.columns = ["action", "timestamp"]

        #the action column is a preview of the one next action given the current sensor values -> shift the
        #action columns upp one row
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
    #column "sensor" contains sensor values, column "sensor (timedelta)" contains time passed since the sensor value
    #changed at the time of the action corresponding to the current row
    sensors = events["sensor"].unique()
    data = pandas.concat([data_for_sensor(sensor, actions) for sensor in sorted(sensors)], axis=1)

    #add action and action timestamp as final columns
    data["action"] = actions["action"]
    data["action_timestamp"] = actions["timestamp"]

    #drop data for actions that correspond to excluded actions
    data_for_excluded = data["action"].isin(excluded_actions)
    data = data[numpy.invert(data_for_excluded)]

    #for the last row, there is no next user actions -> remove that row
    data = data[0:-1]

    data.name = name
    return data


def convert_timedeltas(timedelta_data):
    """
    Convert the numpy timedeltas (which are in nanoseconds) to seconds
    @param original: The original dataset with numpy timedeltas.
    @return: The dataset with timedeltas replaced
    """
    nanoseconds_to_seconds = lambda ts: int(ts.item()/(1000.0*1000.0*1000.0))
    is_missing = lambda ts: str(ts) == "NaT"
    convert_timedelta = lambda ts: numpy.nan if is_missing(ts) else nanoseconds_to_seconds(ts)

    return timedelta_data.applymap(convert_timedelta)


def dataset_to_sklearn(data):
    """
    Convert the dataset into one that can be used by the scikit-learn library [http://scikit-learn.org]

    For a dataset with two binary sensors and 5 instances, the input to this method might look like this:

        s0  s0_timedelta  s1  s1_timedelta action    action_timestamp
    0  NaN           NaT  v0      00:00:25  s0=v0 2013-01-01 00:05:48
    1   v0      00:06:10  v0      00:06:35  s0=v1 2013-01-01 00:11:58
    2   v1      00:00:32  v0      00:07:07  s1=v1 2013-01-01 00:12:30
    3   v1      00:01:10  v1      00:00:38  s1=v0 2013-01-01 00:13:08
    4   v1      00:02:15  v0      00:01:05  s1=v1 2013-01-01 00:14:13

    Since scikit-learn can not handle multi-value nominal attributes the sensor columns must be converted into binary
    features, additionally python timestamps must be converted into seconds:

           (s0, v0)  (s0, v1)  (s1, v0)  (s1, v1)  s0_timedelta  s1_timedelta   action   action_timestamp
    0       NaN       NaN         1         0           NaN            25       s0=v0 2013-01-01 00:05:48
    1         1         0         1         0           370           395       s0=v1 2013-01-01 00:11:58
    2         0         1         1         0            32           427       s1=v1 2013-01-01 00:12:30
    3         0         1         0         1            70            38       s1=v0 2013-01-01 00:13:08
    4         0         1         1         0           135            65       s1=v1 2013-01-01 00:14:13


    The method returns a scikit `Bunch` object that contains:
       name - the name of the dataset
       data - the binarized data as shown above, but without the action and action_timestamp columns, as a numpy 2D array
       target - the contents of the action column, as a numpy array
       features - a list of the column names of the binarized dataset (without action and action_timestamp columns)
       times - the content of the action_timestamp column, as a numpy array
       target_names - a sorted list of all distinct values in the action column

    @param data: The dataset as produced by `load_dataset`.
    @return: The dataset in scikit-learn format.
    """
    #convert a nominal attribute to several binary features, one for each attribute value
    #e.g."door" [open/close] converts to "door=open" (can be 1 or 0) and "door=closed)" (can be 1 or 0)
    def attribute_to_binary(attribute_data):
        attribute_data = attribute_data.dropna()
        if attribute_data.empty:
            return None

        def binary_column_for_value(val):
            return attribute_data.apply(lambda v: 1 if v == val else 0)

        attribute_values = attribute_data.unique()
        binary_columns = [binary_column_for_value(value) for value in attribute_values]
        binary_columns = pandas.concat(binary_columns, axis=1)
        binary_columns.columns = [(attribute_data.name, value) for value in attribute_values]

        return binary_columns

    dataset_name = data.name

    #save actions and action timestamps in separate variables, then drop these columns from the dataset
    targets = data["action"]
    times = data["action_timestamp"]
    data = data.drop(["action", "action_timestamp"], axis=1)

    #seperate columns that contain current sensor values from columns that contain timedelta information
    timedelta_columns = [col for col in data.columns if col.endswith("_timedelta")]
    value_columns = [col for col in data.columns if not col in timedelta_columns]

    #scikit does not support nominal attributes -> convert each attribute to several binary features, one for each value
    binarized_data = pandas.concat([attribute_to_binary(data[attribute]) for attribute in value_columns], axis=1)

    #convert timedelta data from numpy timedeltas to seconds
    binarized_data[timedelta_columns] = convert_timedeltas(data[timedelta_columns])

    return Bunch(name=dataset_name,
                 data=binarized_data.values,
                 target=targets.values,
                 features=binarized_data.columns,
                 times=times.values,
                 target_names=sorted(targets.unique()))


def write_dataset_as_arff(data, path_to_arff):
    """
    Convert the dataset into the arff format that is used by the weka machine learning framework. The resulting file
    can be loaded into dataset and different machine learning algorithms can be tested.
    @param data: The dataset as produced by `load_dataset`.
    @param path_to_arff: The file to which the resulting arff should be written.
    @return: None
    """

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


    def data_as_arff_strings():
        """
        Map each row in the dataset to a string in arff format.
        @return: A list of strings, one for each row in the dataset
        """

        #convert timestamps to format that arff can understand
        timedelta_columns = filter(lambda col: isinstance(data[col][0], numpy.timedelta64), data.columns)
        converted_data = convert_timedeltas(data)

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



