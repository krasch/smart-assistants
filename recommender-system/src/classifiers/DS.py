"""
This module contains functions related to the Dempster-Shafer theory of evidences.
"""

import numpy


def combine_dempsters_rule(masses):
    """
    Implements Dempster's rule of combination. The method makes use of the Moebius transformation and Dempster's rule of
    commonalities for a faster calculation of the combined masses.
    For more information see Kennes, R., Computational aspects of the Mobius transformation of graphs,
    Systems, Man and Cybernetics (Volume:22 ,  Issue: 2 ), http://dx.doi.org/10.1109/21.148425
    @param masses: A list of numpy arrays that all have the same length. Each array contains the mass distribution that
    a source assigns to the available options.
    @return: The combined masses, the conflict between the sources and the theta (masses that can not be assigned to any
    option).
    """

    def mtoq(masses):
        """
        Convert mass vector into commonality vector and calculate theta for this mass vector.
        """
        theta = 1.0 - masses.sum()
        q = masses + theta
        return q, theta

    def qtom(q, theta):
        """
        Convert commonality vector into mass vector.
        """

        masses = q - theta
        return masses

    #convert each mass-vector into a q-vector and calculate theta for each mass vector
    masses_as_q, thetas = zip(*map(mtoq, masses))

    #combine masses by performing element-wise vector multiplication
    combined_masses_as_q = reduce(numpy.multiply, masses_as_q)

    #combine thetas by multiplying the thetas
    combined_theta = reduce(numpy.multiply, thetas)

    #convert masses back from q-form to mass-form
    combined_masses = qtom(combined_masses_as_q, combined_theta)

    #any remaining mass not assigned to specific target or to theta forms the combined conflict
    combined_conflict = 1.0 - combined_masses.sum() - combined_theta
    combined_conflict = max(combined_conflict, 0.0)  #rounding errors sometimes lead to conflict -0.0000000001

    return combined_masses, combined_conflict, combined_theta
