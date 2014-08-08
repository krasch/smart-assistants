# -*- coding: UTF-8 -*-
"""
Evaluate how the recommender system copes with varying interval widths and Delta t_max settings. Details
for this experiment can be found in the paper in Section 6.5 and in the Dissertation in Section 5.5.6.
"""

import sys
sys.path.append("..") 

import pandas

from evaluation.experiment import Experiment
from recsys.classifiers.temporal import TemporalEvidencesClassifier
from recsys.classifiers.binning import initialize_bins
from recsys.dataset import load_dataset

#configuration
data = load_dataset("../datasets/houseA.csv", "../datasets/houseA.config")
intervals_to_test = [#test various settings for delta t_max
                     ("Delta t_max=1200s", initialize_bins(start=0, end=60, width=10) +
                                           initialize_bins(start=60, end=1200, width=30)),
                     ("Delta t_max=120s",  initialize_bins(start=0, end=60, width=10) +
                                           initialize_bins(start=60, end=120, width=30)),
                     ("Delta t_max=60s",   initialize_bins(start=0, end=60, width=10)),
                     ("Delta t_max=30s",   initialize_bins(start=0, end=30, width=10)),
                     ("Delta t_max=10s",   initialize_bins(start=0, end=10, width=10)),
                     #test various interval widths
                     ("all intervals 2s wide",   initialize_bins(start=0, end=300, width=2)),
                     ("all intervals 4s wide",   initialize_bins(start=0, end=300, width=4)),
                     ("all intervals 6s wide",   initialize_bins(start=0, end=300, width=6)),
                     ("all intervals 8s wide",   initialize_bins(start=0, end=300, width=8)),
                     ("all intervals 30s wide",  initialize_bins(start=0, end=300, width=30)),
                     ("all intervals 50s wide",  initialize_bins(start=0, end=300, width=50)),
                     ("all intervals 100s wide", initialize_bins(start=0, end=300, width=100))]

#run 10-fold cross-validation for each of the configured intervals
experiment = Experiment(data)
for (name, bins) in intervals_to_test:
    experiment.add_classifier(TemporalEvidencesClassifier(data.features, data.target_names,
                              bins=bins), name=name)
results = experiment.run(folds=10)

results.print_quality_comparison_at_cutoff(cutoff=1, metrics=["Recall", "Precision", "F1"])
