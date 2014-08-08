# -*- coding: UTF-8 -*-
"""
This module contains functions for dealing with temporal bins in the TemporalClassifier.
"""

import numpy


def initialize_bins(start, end, width):
    """
    Generate a list of interval borders.
    @param start: The left border of the first interval.
    @param end: The left border of the last interval.
    @param width: The width of each interval.
    @return: The list of interval borders.
    """
    return list(range(start + width, end + width, width))


def smooth(x, window_len=9, window='hanning'):
    """
    Perform moving-window smoothing of some value array. This method is copied from numpy cookbook from the numpy
    cookbook, http://www.scipy.org/Cookbook/SignalSmooth
    @param x: The numpy array to be smoothed.
    @param window_len: The size of the window.
    @param window:  The type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
    @return: The smoothed value array.
    """
    window_len = min(window_len, len(x)-1)
    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")
    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")
    if window_len < 3:
        return x
    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is one of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

    s = numpy.r_[2*x[0]-x[window_len-1::-1], x, 2*x[-1]-x[-1:-window_len:-1]]
    if window == 'flat':
        w = numpy.ones(window_len,'d')
    else:
        w = eval('numpy.'+window+'(window_len)')
    y = numpy.convolve(w/w.sum(), s, mode='same')
    return y[window_len:-window_len+1]

