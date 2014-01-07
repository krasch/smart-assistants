from numpy.testing import assert_almost_equal

from src.classifiers.metrics import *
from src.classifiers.bayes import NaiveBayesClassifier
from src.data.kasteren import load_scikit as load_kasteren


"""
Test the calculation of true positives
"""

def test_true_positives():
    targets = ["A"]
    recommendations = [["D", "C", "B", "A", "E", "F", "G"]]
    expected = [0, 0, 0, 1, 1, 1, 1]

    tp = AccuracyMetricsCalculator(targets, recommendations).true_positives("A")
    assert_almost_equal(expected, tp["TP"].values)


def test_true_positives_not_included():
    targets = ["A"]
    recommendations = [["D", "C", "B", "O", "E", "F", "G"]]
    expected = [0, 0, 0, 0, 0, 0, 0]

    tp = AccuracyMetricsCalculator(targets, recommendations).true_positives("A")
    assert_almost_equal(expected, tp["TP"].values)


def test_true_positives_multiple():
    targets = ["A", "B", "A"]
    recommendations = [["D", "C", "B", "A", "E", "F", "G"], #recs for first "A"
                       ["A", "B", "C", "D", "E", "F", "G"], #recs for "B"
                       ["D", "A", "C", "B", "E", "F", "G"]] #recs for second "A"
    expected = [0, 1, 1, 2, 2, 2, 2]

    tp = AccuracyMetricsCalculator(targets, recommendations).true_positives("A")
    assert_almost_equal(expected, tp["TP"].values)


"""
Test the calculation of false  negatives
"""

def test_false_negatives():
    targets = ["A"]
    recommendations = [["D", "C", "B", "A", "E", "F", "G"]]
    expected = [1, 1, 1, 0, 0, 0, 0]

    fn = AccuracyMetricsCalculator(targets, recommendations).false_negatives("A")
    assert_almost_equal(expected, fn["FN"].values)


def test_false_negatives_not_included():
    targets = ["A"]
    recommendations = [["D", "C", "B", "O", "E", "F", "G"]]
    expected = [1, 1, 1, 1, 1, 1, 1]

    fn = AccuracyMetricsCalculator(targets, recommendations).false_negatives("A")
    assert_almost_equal(expected, fn["FN"].values)


def test_false_negatives_multiple():
    targets = ["A", "B", "A"]
    recommendations = [["D", "C", "B", "A", "E", "F", "G"], #recs for first "A"
                       ["A", "B", "C", "D", "E", "F", "G"], #recs for "B"
                       ["D", "A", "C", "B", "E", "F", "G"]] #recs for second "A"
    expected = [2, 1, 1, 0, 0, 0, 0]

    fn = AccuracyMetricsCalculator(targets, recommendations).false_negatives("A")
    assert_almost_equal(expected, fn["FN"].values)

"""
Test the calculation of false positives
"""

def test_false_positives():
    targets = ["B"]
    recommendations = [["D", "C", "B", "A", "E", "F", "G"]]
    expected = [0, 0, 0, 1, 1, 1, 1]

    #true target is "B", how often "A" was recommended instead
    fp = AccuracyMetricsCalculator(targets, recommendations).false_positives("A")
    assert_almost_equal(expected, fp["FP"].values)


def test_false_positives_not_included():
    targets = ["B"]
    recommendations = [["D", "C", "B", "O", "E", "F", "G"]]
    expected = [0, 0, 0, 0, 0, 0, 0]

    fp = AccuracyMetricsCalculator(targets, recommendations).false_positives("A")
    assert_almost_equal(expected, fp["FP"].values)


def test_false_positives_multiple():
    targets = ["B", "A", "B"]
    recommendations = [["D", "C", "B", "A", "E", "F", "G"], #recs for first "B"
                       ["A", "B", "C", "D", "E", "F", "G"], #recs for "A"
                       ["D", "A", "C", "B", "E", "F", "G"]] #recs for second "B"
    expected = [0, 1, 1, 2, 2, 2, 2]

    fp = AccuracyMetricsCalculator(targets, recommendations).false_positives("A")
    assert_almost_equal(expected, fp["FP"].values)


"""
Test the calculation of classification precision
"""

def test_precision():
    targets = ["A"]
    recommendations = [["D", "C", "B", "A", "E", "F", "G"]]
    #true_positives_expected = [0, 0, 0, 1, 1, 1, 1]
    #false_positives_expected = [0, 0, 0, 0, 0, 0, 0]
    expected = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]

    calc = AccuracyMetricsCalculator(targets, recommendations)
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

    calc = AccuracyMetricsCalculator(targets, recommendations)
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

    calc = AccuracyMetricsCalculator(targets, recommendations)
    counts = pandas.concat([calc.true_positives("A"),
                           calc.false_negatives("A"),
                           calc.false_positives("A")],
                           axis=1)
    prec = calc.precision(counts)
    assert_almost_equal(expected, prec["Precision"].values)


"""
Test the calculation of classification recall
"""

def test_recall():
    targets = ["A"]
    recommendations = [["D", "C", "B", "A", "E", "F", "G"]]
    #true_positives_expected = [0, 0, 0, 1, 1, 1, 1]
    #false_negatives_expected = [1, 1, 1, 0, 0, 0, 0]
    expected = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]

    calc = AccuracyMetricsCalculator(targets, recommendations)
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

    calc = AccuracyMetricsCalculator(targets, recommendations)
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

    calc = AccuracyMetricsCalculator(targets, recommendations)
    counts = pandas.concat([calc.true_positives("A"),
                           calc.false_negatives("A"),
                           calc.false_positives("A")],
                           axis=1)
    rec = calc.recall(counts)
    assert_almost_equal(expected, rec["Recall"].values)


"""
Compare overall metrics with known output based on a whole dataset and the Naive Bayes classifier
"""

def test_houseA():
    expected_precision = [0.36389964907400657, 0.35097921410756322, 0.33726522147404076, 0.32783328310438886,
                          0.32350609998195629, 0.31650421215426117, 0.31231855510715401, 0.30519493826462696,
                          0.29979871087041599, 0.29551593405461341, 0.29150527293056733, 0.28660658398334166,
                          0.28251775629715375, 0.27922900853036048, 0.27801361790520279, 0.27550796383161991,
                          0.27466099306467773, 0.27342926821355851, 0.27273588164873097, 0.27221800916720873,
                          0.2721689921161986, 0.27166460270134835, 0.2715840524866433, 0.27152878300894606,
                          0.27150502624152628, 0.27150496777883093]
    expected_recall = [0.50280293229840445, 0.62440707201379908, 0.72229409228115571, 0.78654592496765852,
                       0.84260457093574814, 0.87365243639499779, 0.90426908150064678, 0.93014230271668819,
                       0.95170332039672267, 0.96593359206554552, 0.97714532125916342, 0.98835705045278133,
                       0.99396291504959033, 0.99525657611039242, 0.99568779646399308, 0.99568779646399308,
                       0.99611901681759374, 0.99655023717119451, 0.99655023717119451, 0.99655023717119451,
                       0.99655023717119451, 0.99655023717119451, 0.99655023717119451, 0.99655023717119451,
                       0.99655023717119451, 0.99655023717119451]
    expected_f1 = [0.40969896551479879, 0.40872515566619289, 0.42494674577211028, 0.41795013553355859,
                   0.41402807045377227, 0.40755506280680126, 0.40304176773376232, 0.39634842720545965,
                   0.39155915552074971, 0.38766523517736751, 0.38402205223880148, 0.37854826039224554,
                   0.37434429038914058, 0.37108071816768462, 0.36996547420680476, 0.36801458947276205,
                   0.36727323996045197, 0.36622750241189023, 0.36561611066990368, 0.36516946998665389,
                   0.36510160971503941, 0.36461007637070081, 0.36450029640432352, 0.36442385338288585,
                   0.3643920355519879, 0.36439191983486152]

    #perform classification using NaiveBayes on houseA
    dataset = load_kasteren("houseA")
    cls = NaiveBayesClassifier(dataset.features, dataset.target_names)
    cls = cls.fit(dataset.data, dataset.target)
    results = cls.predict(dataset.data)
    metrics = AccuracyMetricsCalculator(dataset.target, results).calculate()

    assert_almost_equal(metrics["Precision"].values, expected_precision)
    assert_almost_equal(metrics["Recall"].values, expected_recall)
    assert_almost_equal(metrics["F1"].values, expected_f1)