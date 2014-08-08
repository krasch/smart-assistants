# -*- coding: UTF-8 -*-
"""
Plot visualization of user habits, i.e. show which actions typically follow some given user action.
   
Note: the figure for "Frontdoor=Closed" slightly deviates from Figure 1 in the paper and Figure 5.1 in the
dissertation (see paper_experiments.py for bibliographical information). The number of observed actions was reported
correctly in the paper/dissertation, however there was an issue with ordering which actions occur most commonly,
which resulted in "Open cups cupboard" being erroneously included in the figure. Despite this issue, the main point
of the figure still stands: the user has some observable habits after closing the frontdoor.
"""

import sys
sys.path.append("..") 

import pandas

from recsys.classifiers.temporal import TemporalEvidencesClassifier
from recsys.classifiers.binning import initialize_bins
from recsys.dataset import load_dataset
from evaluation import plot
import config

#configuration
data = load_dataset("../datasets/houseA.csv", "../datasets/houseA.config")

#fit classifier to dataset
cls = TemporalEvidencesClassifier(data.features, data.target_names, bins=initialize_bins(0, 300, 10))
cls = cls.fit(data.data, data.target)

#create visualizations of habits around each user action
plot_conf = plot.plot_config(config.plot_directory, sub_dirs=[data.name, "habits"], img_type=config.img_type)
for source in cls.sources.values():
    observations = pandas.DataFrame(source.temporal_counts)
    observations.columns = data.target_names
    observations.index = cls.bins
    plot.plot_observations(source.name(), observations, plot_conf)
    
print "Results can be found in the \"%s\" directory" % config.plot_directory
