# -*- coding: UTF-8 -*-
"""
Create a histogram that compares true positives for different cutoffs using one classifier. Allows to
check which actions are often wrongfully recommended only as second, third, etc alternatives.
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
to_compare = [1, 2, 3, 4]

#run classifier and count true positives
cls = TemporalEvidencesClassifier(data.features, data.target_names)
cls = cls.fit(data.data, data.target)
results = cls.predict(data.data)
results = QualityMetricsCalculator(data.target, results).true_positives_for_all()

#only use the interesting cutoffs
results = results.transpose()[to_compare]
results.columns = ["cutoff=%s" % c for c in results.columns]

conf = plot.plot_config(config.plot_directory, sub_dirs=[data.name], prefix="histogram_cutoffs", img_type=config.img_type)
plot.comparison_histogram(results, conf)
print "Results can be found in the \"%s\" directory" % config.plot_directory
