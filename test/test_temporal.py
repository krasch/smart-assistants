"""
This module tests the proposed classifier (classifiers/temporal.py) with a synthetically generated dataset.
"""

import json

from numpy.testing import assert_array_equal, assert_equal, assert_almost_equal
from numpy import array
import pandas

from src.dataset import load_dataset_as_sklearn
from src.classifiers.temporal import TemporalEvidencesClassifier, Source


#synthetically generated event-list with 5 sensor, 3 nominal values per sensor and 500 events
data_file = "testdata.csv"
#contains the observations the classifier should extract from the testdata
sources_file = "testdata_sources.json"
#contains the recommendations the classifier should generate from the test data
recommendations_file = "testdata_recommendations.json"


def test_train():
    """
    Test that the classifier correctly extracts all observations from the test dataset.
    """
    #train the classifier
    data = load_dataset_as_sklearn(data_file)
    cls = TemporalEvidencesClassifier(data.features, data.target_names)
    cls = cls.fit(data.data, data.target)

    #load expected sources and their observations from json file
    expected_sources = sources_from_json("testdata_sources.json")

    #compare expected with actual sources
    assert_array_equal(sorted(cls.sources.keys()), sorted(expected_sources.keys()),)
    for name in expected_sources.keys():
        assert_source_equal(cls.sources[name], expected_sources[name])


def test_recommend():
    """
    Test that the classifier generates the correct recommendations for the test dataset.
    """

    #train the classifier and calculate recommendations
    data = load_dataset_as_sklearn(data_file)
    cls = TemporalEvidencesClassifier(data.features, data.target_names)
    cls = cls.fit(data.data, data.target)
    actual_recommendations = cls.predict(data.data, include_conflict_theta=True)

    #load expected results from json file
    with open(recommendations_file, 'r') as infile:
        expected_recommendations = json.load(infile)

    #compare expected with actual results
    for actual, expected in zip(actual_recommendations, expected_recommendations):
        assert_recommendations_equal(actual, expected)


"""
Below here are only utility functions.
"""

def sources_to_json(sources, json_file):
    def source_to_dict(source):
        """
        Convert all total and temporal observations of the source into dict, used to enable
        """
        total = pandas.Series(source.total_counts, index=source.targets)
        temporal = pandas.DataFrame(source.temporal_counts, columns=source.targets).transpose()
        return {"total": total.to_dict(), "temporal": temporal.to_dict(),
                "sensor": source.sensor, "value": source.value}

    sources_list = [source_to_dict(sources[name]) for name in sorted(sources.keys())]
    with open(json_file, 'w') as outfile:
        json.dump(sources_list, outfile, indent=4)


def sources_from_json(json_file):
    def source_from_dict(source_dict):
        total = pandas.Series(source_dict["total"])
        temporal = pandas.DataFrame.from_dict(source_dict["temporal"])
        temporal = temporal.reindex_axis(sorted(temporal.columns, key=lambda col: int(col)), axis=1)
        return Source(source_dict["sensor"], source_dict["value"], total, temporal)

    with open(json_file, 'r') as infile:
        sources_list = json.load(infile)

    sources_list = map(source_from_dict, sources_list)
    return {(source.sensor, source.value): source for source in sources_list}


def assert_source_equal(actual, expected):
    assert_equal(actual.sensor, expected.sensor)
    assert_equal(actual.value, expected.value)
    assert_array_equal(actual.targets.values, expected.targets.values)
    assert_almost_equal(actual.total_counts, expected.total_counts)
    assert_equal(len(actual.temporal_counts), len(expected.temporal_counts))
    assert_almost_equal(array(actual.temporal_counts), array(expected.temporal_counts))


def assert_recommendations_equal(actual, expected):
    actual_recommendations, actual_conflict, actual_theta = actual
    expected_recommendations, expected_conflict, expected_theta = actual
    assert_array_equal(actual_recommendations, expected_recommendations)
    assert_almost_equal(actual_conflict, expected_conflict)
    assert_almost_equal(actual_theta, expected_theta)




