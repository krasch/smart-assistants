"""
This module tests if metrics such as precision and recall are calculated correctly.
"""

from numpy.testing import assert_almost_equal

from src.evaluation.metrics import *


"""
Following three methods test the calculation of true positives
"""
def test_true_positives():
    targets = ["A"]
    recommendations = [["D", "C", "B", "A", "E", "F", "G"]]
    expected = [0, 0, 0, 1, 1, 1, 1]

    tp = QualityMetricsCalculator(targets, recommendations).true_positives("A")
    assert_almost_equal(expected, tp["TP"].values)


def test_true_positives_not_included():
    targets = ["A"]
    recommendations = [["D", "C", "B", "O", "E", "F", "G"]]
    expected = [0, 0, 0, 0, 0, 0, 0]

    tp = QualityMetricsCalculator(targets, recommendations).true_positives("A")
    assert_almost_equal(expected, tp["TP"].values)


def test_true_positives_multiple():
    targets = ["A", "B", "A"]
    recommendations = [["D", "C", "B", "A", "E", "F", "G"], #recs for first "A"
                       ["A", "B", "C", "D", "E", "F", "G"], #recs for "B"
                       ["D", "A", "C", "B", "E", "F", "G"]] #recs for second "A"
    expected = [0, 1, 1, 2, 2, 2, 2]

    tp = QualityMetricsCalculator(targets, recommendations).true_positives("A")
    assert_almost_equal(expected, tp["TP"].values)


"""
Following three methods test the calculation of false  negatives
"""

def test_false_negatives():
    targets = ["A"]
    recommendations = [["D", "C", "B", "A", "E", "F", "G"]]
    expected = [1, 1, 1, 0, 0, 0, 0]

    fn = QualityMetricsCalculator(targets, recommendations).false_negatives("A")
    assert_almost_equal(expected, fn["FN"].values)


def test_false_negatives_not_included():
    targets = ["A"]
    recommendations = [["D", "C", "B", "O", "E", "F", "G"]]
    expected = [1, 1, 1, 1, 1, 1, 1]

    fn = QualityMetricsCalculator(targets, recommendations).false_negatives("A")
    assert_almost_equal(expected, fn["FN"].values)


def test_false_negatives_multiple():
    targets = ["A", "B", "A"]
    recommendations = [["D", "C", "B", "A", "E", "F", "G"], #recs for first "A"
                       ["A", "B", "C", "D", "E", "F", "G"], #recs for "B"
                       ["D", "A", "C", "B", "E", "F", "G"]] #recs for second "A"
    expected = [2, 1, 1, 0, 0, 0, 0]

    fn = QualityMetricsCalculator(targets, recommendations).false_negatives("A")
    assert_almost_equal(expected, fn["FN"].values)

"""
Following three methods test the calculation of false positives
"""

def test_false_positives():
    targets = ["B"]
    recommendations = [["D", "C", "B", "A", "E", "F", "G"]]
    expected = [0, 0, 0, 1, 1, 1, 1]

    #true target is "B", how often "A" was recommended instead
    fp = QualityMetricsCalculator(targets, recommendations).false_positives("A")
    assert_almost_equal(expected, fp["FP"].values)


def test_false_positives_not_included():
    targets = ["B"]
    recommendations = [["D", "C", "B", "O", "E", "F", "G"]]
    expected = [0, 0, 0, 0, 0, 0, 0]

    fp = QualityMetricsCalculator(targets, recommendations).false_positives("A")
    assert_almost_equal(expected, fp["FP"].values)


def test_false_positives_multiple():
    targets = ["B", "A", "B"]
    recommendations = [["D", "C", "B", "A", "E", "F", "G"], #recs for first "B"
                       ["A", "B", "C", "D", "E", "F", "G"], #recs for "A"
                       ["D", "A", "C", "B", "E", "F", "G"]] #recs for second "B"
    expected = [0, 1, 1, 2, 2, 2, 2]

    fp = QualityMetricsCalculator(targets, recommendations).false_positives("A")
    assert_almost_equal(expected, fp["FP"].values)

"""
Following three methods the the calculation of classification precision
"""

def test_precision():
    targets = ["A"]
    recommendations = [["D", "C", "B", "A", "E", "F", "G"]]
    #true_positives_expected = [0, 0, 0, 1, 1, 1, 1]
    #false_positives_expected = [0, 0, 0, 0, 0, 0, 0]
    expected = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]

    calc = QualityMetricsCalculator(targets, recommendations)
    counts = pandas.concat([calc.true_positives("A"),
                           calc.false_negatives("A"),
                           calc.false_positives("A")],
                           axis=1)
    prec = calc.precision(counts)
    assert_almost_equal(expected, prec["Precision"].values)


def test_precision_not_included():
    targets = ["A"]
    recommendations = [["D", "C", "B", "O", "E", "F", "G"]]
    #true_positives_expected = [0, 0, 0, 0, 0, 0, 0]
    #false_positives_expected = [0, 0, 0, 0, 0, 0, 0]
    expected = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    calc = QualityMetricsCalculator(targets, recommendations)
    counts = pandas.concat([calc.true_positives("A"),
                           calc.false_negatives("A"),
                           calc.false_positives("A")],
                           axis=1)
    prec = calc.precision(counts)
    assert_almost_equal(expected, prec["Precision"].values)


def test_precision_multiple():
    targets = ["A", "B", "A"]
    recommendations = [["D", "C", "B", "A", "E", "F", "G"], #recs for first "A"
                       ["A", "B", "C", "D", "E", "F", "G"], #recs for "B"
                       ["D", "A", "C", "B", "E", "F", "G"]] #recs for second "A"
    #true_positives_expected = [0, 1, 1, 2, 2, 2, 2]
    #false_positives_expected = [1, 1, 1, 1, 1, 1, 1]
    expected = [0.0, 0.5, 0.5, 2.0/3.0, 2.0/3.0, 2.0/3.0, 2.0/3.0]

    calc = QualityMetricsCalculator(targets, recommendations)
    counts = pandas.concat([calc.true_positives("A"),
                           calc.false_negatives("A"),
                           calc.false_positives("A")],
                           axis=1)
    prec = calc.precision(counts)
    assert_almost_equal(expected, prec["Precision"].values)


"""
Following three methods test the calculation of classification recall
"""

def test_recall():
    targets = ["A"]
    recommendations = [["D", "C", "B", "A", "E", "F", "G"]]
    #true_positives_expected = [0, 0, 0, 1, 1, 1, 1]
    #false_negatives_expected = [1, 1, 1, 0, 0, 0, 0]
    expected = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]

    calc = QualityMetricsCalculator(targets, recommendations)
    counts = pandas.concat([calc.true_positives("A"),
                           calc.false_negatives("A"),
                           calc.false_positives("A")],
                           axis=1)
    rec = calc.recall(counts)
    assert_almost_equal(expected, rec["Recall"].values)


def test_recall_not_included():
    targets = ["A"]
    recommendations = [["D", "C", "B", "O", "E", "F", "G"]]
    #true_positives_expected = [0, 0, 0, 0, 0, 0, 0]
    #false_negatives_expected = [0, 0, 0, 0, 0, 0, 0]
    expected = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    calc = QualityMetricsCalculator(targets, recommendations)
    counts = pandas.concat([calc.true_positives("A"),
                           calc.false_negatives("A"),
                           calc.false_positives("A")],
                           axis=1)
    rec = calc.recall(counts)
    assert_almost_equal(expected, rec["Recall"].values)


def test_recall_multiple():
    targets = ["A", "B", "A"]
    recommendations = [["D", "C", "B", "A", "E", "F", "G"], #recs for first "A"
                       ["A", "B", "C", "D", "E", "F", "G"], #recs for "B"
                       ["D", "A", "C", "B", "E", "F", "G"]] #recs for second "A"
    #true_positives_expected = [0, 1, 1, 2, 2, 2, 2]
    #false_negatives_expected = [2, 1, 1, 0, 0, 0, 0]
    expected = [0.0, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0]

    calc = QualityMetricsCalculator(targets, recommendations)
    counts = pandas.concat([calc.true_positives("A"),
                           calc.false_negatives("A"),
                           calc.false_positives("A")],
                           axis=1)
    rec = calc.recall(counts)
    assert_almost_equal(expected, rec["Recall"].values)




