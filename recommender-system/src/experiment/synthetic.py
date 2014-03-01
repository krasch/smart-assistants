# -*- coding: UTF-8 -*-
"""
This module contains synthetic dataset generators which can be used to evaluate the scalability of the classifiers.
"""

import random
from datetime import datetime, timedelta

import pandas
import numpy

from dataset import events_to_dataset, dataset_to_sklearn
from classifiers.temporal import TemporalEvidencesClassifier, Source

def generate_random_events(sensors, num_events, at_least_one_per_setting=False):
    """
    Generates a sequence of random sensor events, each described by "timestamp", "sensor", "value"
    @param sensors: A dictionary with sensor names as keys and possible sensor settings as values.
    @param num_events: The number of random events to generate.
    @param at_least_one_per_setting: Whether the event-list should at least contain one event per possible sensor
    setting, If this parameter is true and num_events < #possible_settings then #possible_settings
    will be returned,
    @return: A pandas dataframe with three columns "timestamp", "sensor", "value".
    """

    #there should be between 1 and 400 seconds between each event, starting from 1. January 2013, 00:00
    random_timedelta = lambda: timedelta(seconds=random.randint(1, 400))
    current_timestamp = [datetime(year=2013, month=1, day=1)]

    #generate an event with random sensor and value, avoiding that duplicate values for a sensor appear consecutively
    current_sensor_settings = {s: numpy.nan for s in sensors}
    def generate_event(sensor=None, value=None):
        sensor = sensor or random.choice(sensors.keys())
        possible_values = [v for v in sensors[sensor] if not current_sensor_settings[sensor] == v]
        current_sensor_settings[sensor] = value or random.choice(possible_values)
        current_timestamp[0] += random_timedelta()
        return current_timestamp[0], sensor, current_sensor_settings[sensor]

    #if necessary, generate at least one event for each possible setting
    events = []
    if at_least_one_per_setting:
        events = [generate_event(sensor, value) for sensor in sorted(sensors.keys())
                                                for value in sensors[sensor]]

    #generate remaining events randomly
    events += [generate_event() for e in range(num_events - len(events))]

    return pandas.DataFrame(events, columns=["timestamp", "sensor", "value"])


def generate_synthetic_dataset(num_sensors, nominal_values_per_sensor, num_instances):
    """
    Generate a random dataset in the format required by scikit-learn, see dataset.dataset_to_sklearn for more details.
    @param num_sensors: Number of different sensors in the dataset.
    @param nominal_values_per_sensor: Number of possible settings for each sensor.
    @param num_instances: Number of data instances in the dataset.
    @return: A random dataset in scikit-learn format.
    """

    #generate the desired number of sensors, all have the same number of possible settings
    sensor_settings = set("v%d" % id for id in range(nominal_values_per_sensor))
    sensor_name = lambda id: "s%d" % id
    sensors = {sensor_name(id): sensor_settings for id in range(num_sensors)}

    events = generate_random_events(sensors, num_instances)
    dataset = events_to_dataset(events, name="synthetic", excluded_sensors=[], excluded_actions=[])

    return dataset_to_sklearn(dataset)


def generate_trained_classifier(num_sensors, nominal_values_per_sensor, num_test_instances):
    """
    If we just want to evaluate the scalability of service recommendations, we can skip the learning phase and fill
    the classifier with random observations. This method creates such a "trained" classifier and returns also a
    compatible test dataset for running the scalability experiments.
    @param num_sensors: Number of different sensors in the dataset.
    @param nominal_values_per_sensor: Number of possible settings for each sensor.
    @param num_test_instances: Number of data instances in the test dataset.
    @return: A tuple of trained classifier and a compatible test dataset.
    """

    def generate_classifier(sensors):
        #create a source for the setting sensor=value, fill it with random observations
        def create_source(sensor, value, num_bins):
            random_observations = lambda: pandas.Series(numpy.random.randint(0, 100, len(targets)), index=targets)
            temporal = pandas.concat([random_observations() for b in range(num_bins)], axis=1)
            total = random_observations()
            return Source(sensor, value, total, temporal)

        #initialize the classifier
        all_settings = [(sensor, value) for sensor in sensors.keys() for value in sensors[sensor]]
        features = sorted(all_settings) + ["%s_timedelta" % sensor for sensor in sorted(sensors.keys())]
        targets = ["%s=%s" % (sensor, value) for sensor, value in sorted(all_settings)]
        cls = TemporalEvidencesClassifier(features, targets)

        #create a random sources for each possible setting
        cls.sources = {(sensor, value): create_source(sensor, value, len(cls.binning_method.bins))
                       for sensor, value in all_settings}
        cls.max_total = max(source.total_counts.sum() for source in cls.sources.values())
        cls.max_temporal = max(source.max_temporal() for source in cls.sources.values())

        return cls

    def generate_test_data(sensors):
        events = generate_random_events(sensors, num_test_instances, at_least_one_per_setting=True)
        data = events_to_dataset(events, name="synthetic", excluded_sensors=[], excluded_actions=[])
        return dataset_to_sklearn(data)


    #generate the desired number of sensors, all have the same number of possible settings
    sensor_settings = set("v%d" % id for id in range(nominal_values_per_sensor))
    sensor_name = lambda id: "s%d" % id
    sensors = {sensor_name(id): sensor_settings for id in range(num_sensors)}

    #generate the classifier
    cls = generate_classifier(sensors)
    #generate test instances that have the same features and targets that where used to generate the classifier
    data = generate_test_data(sensors)

    return cls, data

