from collections import defaultdict
from numpy import array,mean
import numpy

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

"""
@profile
def simple_conjunctive_combiner(distributions):
    def calculate_theta(masses):
        return 1.0-masses.sum()

    def combine_two((masses1,theta1),(masses2,theta2)):
        #masses=masses1*masses2+masses1*theta2+theta1*masses2
        masses=masses1*(masses2+theta2)+masses2*theta1
        theta=theta1*theta2
        #conflict=1.0-sum(masses)-theta
        return masses,theta
    
    comb_masses,comb_theta=(distributions[0],calculate_theta(distributions[0]))
    for dis in distributions[1:]:
        comb_masses,comb_theta=combine_two((comb_masses,comb_theta),(dis,calculate_theta(dis)))
    comb_conflict=1.0-comb_masses.sum()-comb_theta
    return comb_masses,comb_conflict,comb_theta
"""

#using moebius vector
#@profile
def simple_conjunctive_combiner(distributions):
    def mtoq(masses):
        theta=1.0-masses.sum()
        q=masses+theta
        return q,theta

    def qtom(q,theta):
        masses=q-theta
        return masses
  
    comb_q,comb_theta=mtoq(distributions[0])
    for dis in distributions[1:]:
        q,theta=mtoq(dis)
        comb_q*=q
        comb_theta*=theta
        
    comb_masses=qtom(comb_q,comb_theta)
    comb_conflict=1.0-comb_masses.sum()-comb_theta
    return comb_masses,comb_conflict,comb_theta

def normalized_random_vector(length):
    v=numpy.random.randint(1,10,length)
    sum_=v.sum()+10
    return v/float(sum_)

len_distributions=2500
len_q=2500
distributions=[normalized_random_vector(len_q) for i in range(len_distributions)]
#distributions=[numpy.array([0.3,0.4,0.2]),numpy.array([0.1,0.1,0.4])]
simple_conjunctive_combiner(distributions)
