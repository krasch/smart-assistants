# -*- coding: UTF-8 -*-
"""
Evaluates the benefit of dynamic cutoff methods, i.e. show less recommendations if uncertainty and conflict are low.
Further details for this experiment can be found in the paper in Section 6.6 and the dissertation in Section 5.5.7
"""

import sys
sys.path.append("..") 

import pandas

from evaluation.experiment import Experiment
from evaluation.metrics import quality_metrics
from recsys.classifiers.temporal import TemporalEvidencesClassifier, configure_dynamic_cutoff
from recsys.dataset import load_dataset


#configuration
data = load_dataset("../datasets/houseA.csv", "../datasets/houseA.config")
methods_to_test = [("Fixed cutoff", None),
                   ("dynamic cutoff=4", configure_dynamic_cutoff(1.0, 0.4, 4)),
                   ("dynamic cutoff=2", configure_dynamic_cutoff(1.0, 0.4, 2))]

#run all configured cutoffs with 10-fold cross-validation
experiment = Experiment(data)
for name, method in methods_to_test:
    experiment.add_classifier(TemporalEvidencesClassifier(data.features, data.target_names,
                              postprocess=method), name=name)
results = experiment.run(folds=10)

#print results
pandas.set_option('expand_frame_repr', False)
pandas.set_option('max_columns', 4)
print "Maximum 5 recommendations"
results.print_quality_comparison_at_cutoff(cutoff=5, metrics=quality_metrics)
print "Maximum 10 recommendations"
results.print_quality_comparison_at_cutoff(cutoff=10, metrics=quality_metrics)
