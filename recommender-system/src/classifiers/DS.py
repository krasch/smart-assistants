from collections import defaultdict
from numpy import array,mean
import numpy

import pandas

from profilehooks import profile

def all_elements(distributions):
    elements=set()
    for (masses,conflict,theta) in distributions:
        elements=elements|set(masses.keys())
    return list(elements)

def arithmetic_mean_combiner(distributions):
    result_theta=mean(array([theta for (masses,conflict,theta) in distributions]))
    result_conflict=mean(array([conflict for (masses,conflict,theta) in distributions]))
    result_masses=dict()
    for e in all_elements(distributions):
        result_masses[e]=mean(array([masses.get(e,0.0) for (masses,conflict,theta) in distributions]))
    return (result_masses,result_conflict,result_theta)

#using moebius vector
#@profile
def simple_conjunctive_combiner(distributions):

    def mtoq(masses):
        theta = 1.0 - masses.sum()
        q = masses + theta
        return q, theta

    def qtom(q, theta):
        masses = q - theta
        return masses

    combined_q, combined_theta = mtoq(distributions[0])
    for dis in distributions[1:]:
        q, theta = mtoq(dis)
        combined_q *= q
        combined_theta *= theta

    combined_masses = qtom(combined_q, combined_theta)
    combined_conflict = max(1.0 - combined_masses.sum() - combined_theta, 0.0)

    return combined_masses, combined_conflict, combined_theta


def normalized_random_vector(length):
    v=numpy.random.randint(1,10,length)
    sum_=v.sum()+10
    return numpy.array(v/float(sum_))


#len_distributions = 2500
#len_q = 2500
#distributions=[normalized_random_vector(len_q) for i in range(len_distributions)]

#distributions = numpy.array(distributions)
#distributions.sum(axis=1)
#distributions = [pandas.Series([0.3, 0.4, 0.2]), pandas.Series([0.1, 0.1, 0.4])]
#distributions = pandas.concat(distributions, axis=1)
#simple_conjunctive_combiner(distributions)


#correct:
# (array([ 0.16,  0.21,  0.2 ]), 0.39000000000000012, 0.040000000000000029)
#(array([ 0.4,  0.5,  0.3]), 0.10000000000000009)
#(array([ 0.5,  0.5,  0.8]), 0.39999999999999991)
#combined: [ 0.2   0.25  0.24]

"""
*** PROFILER RESULTS ***
simple_conjunctive_combiner (/home/kat/arbeit/smart-assistants/recommender-system/src/classifiers/DS.py:42)
function called 1 times

         10006 function calls in 0.106 seconds

   Ordered by: cumulative time, internal time, call count

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.023    0.023    0.106    0.106 DS.py:42(simple_conjunctive_combiner)
     2500    0.033    0.000    0.083    0.000 DS.py:44(mtoq)
     2501    0.003    0.000    0.050    0.000 {method 'sum' of 'numpy.ndarray' objects}
     2501    0.004    0.000    0.047    0.000 _methods.py:23(_sum)
     2501    0.043    0.000    0.043    0.000 {method 'reduce' of 'numpy.ufunc' objects}
        1    0.000    0.000    0.000    0.000 DS.py:49(qtom)
        1    0.000    0.000    0.000    0.000 {method 'disable' of '_lsprof.Profiler' objects}
        0    0.000             0.000          profile:0(profiler)
"""