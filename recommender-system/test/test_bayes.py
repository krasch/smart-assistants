"""
This module tests the implementation of the Naive Bayes classifier.
"""

from numpy.testing import assert_almost_equal

from src.experiment.metrics import *
from src.classifiers.bayes import NaiveBayesClassifier
from src.dataset import load_dataset_as_sklearn


houseA_csv = "../datasets/houseA.csv"
houseA_config = "../datasets/houseA.config"


def test_houseA():
    """
    Check if recommendation results of the Naive Bayes classifier are as expected for the houseA dataset.
    """

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
    dataset = load_dataset_as_sklearn(houseA_csv, houseA_config)
    cls = NaiveBayesClassifier(dataset.features, dataset.target_names)
    cls = cls.fit(dataset.data, dataset.target)
    results = cls.predict(dataset.data)
    metrics = QualityMetricsCalculator(dataset.target, results).calculate()

    assert_almost_equal(metrics["Precision"].values, expected_precision, decimal=3)
    assert_almost_equal(metrics["Recall"].values, expected_recall, decimal=3)
    assert_almost_equal(metrics["F1"].values, expected_f1, decimal=3)