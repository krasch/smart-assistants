# -*- coding: UTF-8 -*-
"""
Evaluate how the proposed recommendation algorithm scales for larger datasets with many sensors and several nominal
values per sensor. The data for this experiment is synthetically generated. Details for the experiment can be found
in the paper in Section 6.7 and in the dissertation in Section 5.5.9,
"""

import timeit

#setup necessary to run timeit function
setup = '''
import sys
sys.path.append("..") 
from synthetic import generate_trained_classifier
from scalability import num_sensors, nominal_values_per_sensor, num_instances
cls, test_data = generate_trained_classifier(num_sensors=num_sensors,\
                                         nominal_values_per_sensor=nominal_values_per_sensor, \
                                         num_test_instances=num_instances)
'''
#evaluation parameters
global num_instances, num_sensors, nominal_values_per_sensor
num_instances = 1000
num_sensors = 100
nominal_values_per_sensor = 5
seconds_to_milliseconds = lambda seconds: seconds*1000.0

#evaluate
timer = timeit.Timer('cls.predict(test_data.data)', setup=setup)
test_time = seconds_to_milliseconds(min(timer.repeat(repeat=3, number=1)))
test_time_per_instance = test_time / num_instances
#print "Total testing time %.4f [ms]" %test_time
print "Testing time per instance %.4f [ms]" % test_time_per_instance
