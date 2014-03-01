"""
This module tests the implementation of Dempster-Shafer theory.
"""

import numpy
from numpy.testing import assert_almost_equal
from profilehooks import profile

from src.classifiers.DS import combine_dempsters_rule

def test_simple_mass_distribution():
    distributions = [numpy.array([0.3, 0.4, 0.2]), numpy.array([0.1, 0.1, 0.4])]
    expected_combined = numpy.array([ 0.16,  0.21,  0.2 ])
    expected_conflict = 0.39
    expected_theta = 0.04

    combined, conflict, theta = combine_dempsters_rule(distributions)
    assert_almost_equal(combined, expected_combined)
    assert_almost_equal(conflict, expected_conflict)
    assert_almost_equal(theta, expected_theta)

@profile
def test_runtime():
    def normalized_random_vector(length):
        v=numpy.random.randint(1,10,length)
        sum_=v.sum()+10
        return numpy.array(v/float(sum_))


    len_distributions = 2500
    len_q = 2500
    distributions=[normalized_random_vector(len_q) for i in range(len_distributions)]
    combine_dempsters_rule(distributions)
