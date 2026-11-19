import pandas as pd

from sklearn.datasets import load_iris

from automl.meta_feature_extractor import DatasetMetaFeatureExtractor


def test_meta_features_are_generated_for_classification():
    iris = load_iris(as_frame=True)
    df = iris.frame.copy()
    df["target"] = iris.target_names[iris.target]

    extractor = DatasetMetaFeatureExtractor(df, "target")
    features = extractor.extract_all()

    assert isinstance(features, dict)
    assert features["problem_type"] == "classification"
    assert features["n_samples"] == len(df)
    assert features["n_features"] == df.shape[1] - 1
    assert features["n_numeric_features"] >= 1
    assert features["n_categorical_features"] >= 0
    assert features["feature_sample_ratio"] > 0
    assert features["duplicate_ratio"] >= 0
    assert features["target_cardinality"] == 3
    assert "class_imbalance_ratio" in features
    assert "target_distribution" in features
    assert "class_distribution" in features
    assert all(pd.notna(value) for value in features.values() if isinstance(value, (int, float)))


def test_meta_features_return_expected_keys():
    df = pd.DataFrame(
        {
            "a": [1.0, 2.0, 3.0, 4.0],
            "b": ["x", "x", "y", "y"],
            "c": [10, 11, 12, 13],
            "target": ["yes", "yes", "no", "no"],
        }
    )

    extractor = DatasetMetaFeatureExtractor(df, "target")
    features = extractor.extract_all()

    expected_keys = {
        "n_samples",
        "n_features",
        "n_numeric_features",
        "n_categorical_features",
        "missing_ratio",
        "duplicate_ratio",
        "target_cardinality",
        "feature_sample_ratio",
        "problem_type",
        "class_imbalance_ratio",
        "target_distribution",
    }

    assert expected_keys.issubset(features.keys())
    assert 0 <= features["missing_ratio"] <= 1
    assert 0 <= features["duplicate_ratio"] <= 1
    assert features["n_numeric_features"] == 2
    assert features["n_categorical_features"] == 1


def test_numeric_classification_target_is_not_counted_as_a_feature():
    df = pd.DataFrame(
        {
            "numeric_feature": [1.0, 2.0, 3.0, 4.0],
            "target": [0, 1, 0, 1],
        }
    )

    features = DatasetMetaFeatureExtractor(df, "target").extract_all()

    assert features["problem_type"] == "classification"
    assert features["n_features"] == 1
    assert features["n_numeric_features"] == 1
    assert features["numeric_feature_ratio"] == 1.0
