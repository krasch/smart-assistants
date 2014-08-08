# -*- coding: UTF-8 -*-
"""
Scatter conflict versus uncertainty at different cutoffs to find regions of uncertainty/conflict where the algorithm
is more/less successful. Further details for this experiment can be found in the paper in Section 6.6 and the
dissertation in Section 5.5.7.
"""

import sys
sys.path.append("..") 

import pandas

from recsys.classifiers.temporal import TemporalEvidencesClassifier
from recsys.dataset import load_dataset
from evaluation.metrics import results_as_dataframe
from evaluation import plot
import config


#configuration
data = load_dataset("../datasets/houseA.csv", "../datasets/houseA.config")

#run the classifier on the whole dataset
cls = TemporalEvidencesClassifier(data.features, data.target_names)
cls = cls.fit(data.data, data.target)
results = cls.predict(data.data, include_conflict_theta=True)

#extract conflict and uncertainty and convert recommendations to pandas representation
recommendations, conflict, uncertainty = zip(*results)
results = results_as_dataframe(data.target, list(recommendations))

#for each row, mark correct recommendations with "1", false recommendations with "0"
find_matches_in_row = lambda row: [1 if col == row.name else 0 for col in row]
results = results.apply(find_matches_in_row, axis=1)

#set uncertainty and conflict as multi-index
results.index = pandas.MultiIndex.from_tuples(zip(conflict, uncertainty),
                                              names=["Conflict", "Uncertainty"])

#found_within: the correct service was found within X recommendations
#-> apply cumulative sum on each row so that the "1" marker is set for all columns after it first appears
found_within = results.cumsum(axis=1)
#create one plot for each cutoff
conf = plot.plot_config(config.plot_directory, sub_dirs=[data.name, "conflict-uncertainty"],
                        prefix="found_within_", img_type=config.img_type)
plot.conflict_uncertainty_scatter(found_within, conf)

#not found withing: the correct service was not found within X recommendations, is the reverse of found_within
not_found_within = found_within.apply(lambda col: 1-col)
#create one plot for each cutoff
conf = plot.plot_config(config.plot_directory, sub_dirs=[data.name, "conflict-uncertainty"],
                        prefix="not_found_within_", img_type=config.img_type)
plot.conflict_uncertainty_scatter(not_found_within, conf)

print "Results can be found in the \"%s\" directory" % config.plot_directory
