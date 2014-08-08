# -*- coding: UTF-8 -*-
"""
Print a confusion matrix: list for each action how often each service was recommended
"""

import sys
sys.path.append("..") 

import pandas

from recsys.classifiers.temporal import TemporalEvidencesClassifier
from recsys.dataset import load_dataset
from evaluation.metrics import QualityMetricsCalculator


#configuration
data = load_dataset("../datasets/houseA.csv", "../datasets/houseA.config")
#data = load_dataset("../datasets/houseB.csv", "../datasets/houseB.config")

#run the classifier on the whole dataset and calculate confusion matrix
cls = TemporalEvidencesClassifier(data.features, data.target_names)
cls = cls.fit(data.data, data.target)
results = cls.predict(data.data)
matrix = QualityMetricsCalculator(data.target, results).confusion_matrix()

#format confusion matrix for pretty printing
letters = list(map(chr, list(range(97, 123))))+list(map(chr, list(range(65, 91))))
action_to_letter = {action: letter for action, letter in zip(matrix.index, letters)}
matrix.columns = [action_to_letter[action] for action in matrix.columns]
matrix.index = ["(%s) %s" % (action_to_letter[action], action) for action in matrix.index]
matrix.index.name = "Actual action"

pandas.set_option('expand_frame_repr', False)
pandas.set_option('max_columns', 40)
print matrix
