# -*- coding: UTF-8 -*-
"""
Create a histogram that compares true positives for different classifiers/classifier settings. Allows to
check how well the classifiers succeed with predicting each of the actions.
"""

import sys
sys.path.append("..") 

import pandas

from recsys.classifiers.temporal import TemporalEvidencesClassifier
from recsys.classifiers.bayes import NaiveBayesClassifier
from recsys.dataset import load_dataset
from evaluation import plot
from evaluation.metrics import QualityMetricsCalculator
import config


#configuration
data = load_dataset("../datasets/houseA.csv", "../datasets/houseA.config")
classifiers = [NaiveBayesClassifier(data.features, data.target_names),
              TemporalEvidencesClassifier(data.features, data.target_names)]

#run the experiment using full dataset as training and as test data
results = []
for cls in classifiers:
    cls = cls.fit(data.data, data.target)
    r = cls.predict(data.data)
    r = QualityMetricsCalculator(data.target, r)
    results.append(r.true_positives_for_all())

#want for each classifier result only the measurements for cutoff=1
results = [r.loc[1] for r in results]
results = pandas.concat(results, axis=1)
results.columns = [cls.name for cls in classifiers]

plot_conf = plot.plot_config(config.plot_directory, sub_dirs=[data.name], prefix="histogram_classifiers", img_type=config.img_type)
plot.comparison_histogram(results, plot_conf)
print "Results can be found in the \"%s\" directory" % config.plot_directory
