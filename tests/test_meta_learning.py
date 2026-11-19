import pandas as pd
import pytest

from meta_learning.build_meta_dataset import build_meta_dataset, choose_best_trial
from meta_learning.common import prepare_splits
from meta_learning import configuration_search
from meta_learning.results import summarize_results
from meta_learning.train_meta_model import predict_configuration, train_meta_model


def test_prepare_splits_reserves_validation_and_test_data():
    dataset = pd.DataFrame(
        {
            "feature": range(100),
            "target": ["a", "b"] * 50,
        }
    )

    splits = prepare_splits(dataset, "target", random_state=11)

    assert len(splits["y_train"]) == 64
    assert len(splits["y_validation"]) == 16
    assert len(splits["y_test"]) == 20
    assert splits["problem_type"] == "classification"
    assert splits["raw_meta_features"]["N"] == (
        len(splits["y_train"]) + len(splits["y_validation"])
    )


def test_prepare_splits_extracts_meta_features_from_training_rows_only():
    dataset = pd.DataFrame(
        {
            "feature": range(100),
            "target": ["a"] * 80 + ["b"] * 20,
        }
    )

    splits = prepare_splits(dataset, "target", random_state=11)

    development_targets = pd.concat([splits["y_train"], splits["y_validation"]])
    assert splits["raw_meta_features"]["N"] == len(development_targets)
    assert splits["raw_meta_features"]["F"] == 1
    assert splits["raw_meta_features"]["P"] == "classification"


def test_meta_dataset_build_resumes_successful_configurations(tmp_path, monkeypatch):
    dataset_path = tmp_path / "dataset.csv"
    pd.DataFrame(
        {"feature": range(100), "target": ["a", "b"] * 50}
    ).to_csv(dataset_path, index=False)
    calls = []

    def fake_run_tpot(splits, **configuration):
        calls.append(configuration)
        return {
            "accuracy": 0.8,
            "f1_score": 0.8,
            "mae": None,
            "r2_score": None,
            "runtime_seconds": 1.0,
            "success": True,
            "error_message": None,
        }

    monkeypatch.setattr("meta_learning.build_meta_dataset.run_tpot", fake_run_tpot)
    output_dir = tmp_path / "experiment"
    dataset_spec = f"{dataset_path}=target"

    build_meta_dataset([dataset_spec], output_dir=output_dir)
    assert len(calls) == 5

    build_meta_dataset([dataset_spec], output_dir=output_dir)

    assert len(calls) == 5
    trials = pd.read_csv(output_dir / "configuration_trials.csv")
    assert len(trials) == 5


def test_meta_model_predicts_an_observed_configuration(tmp_path):
    rows = []
    for index, (population, generations) in enumerate(((3, 1), (5, 2), (10, 3))):
        rows.append(
            {
                "dataset_name": f"data-{index}",
                "n_samples": 50 + index * 20,
                "n_features": 4 + index,
                "n_numeric_features": 4 + index,
                "n_categorical_features": 0,
                "numeric_feature_ratio": 1.0,
                "categorical_feature_ratio": 0.0,
                "missing_ratio": 0.0,
                "duplicate_ratio": 0.0,
                "target_cardinality": 2,
                "feature_sample_ratio": 0.08,
                "dataset_dimensionality": (50 + index * 20) * (4 + index),
                "average_numeric_feature_std": 1.0,
                "average_numeric_feature_mean": 0.0,
                "average_numeric_feature_missing_ratio": 0.0,
                "class_imbalance_ratio": 1.0,
                "problem_type": "classification",
                "selected_population_size": population,
                "selected_generations": generations,
            }
        )
    model_path = tmp_path / "meta_model.joblib"

    train_meta_model(pd.DataFrame(rows), model_path)
    prediction = predict_configuration(rows[0], model_path)

    assert (prediction["population_size"], prediction["generations"]) in {
        (3, 1),
        (5, 2),
        (10, 3),
    }


def test_predict_configuration_uses_model_prediction_before_fallback(tmp_path):
    rows = []
    for index, (population, generations) in enumerate(((3, 1), (5, 2), (10, 3))):
        rows.append(
            {
                "dataset_name": f"data-{index}",
                "n_samples": 50 + index * 20,
                "n_features": 4 + index,
                "n_numeric_features": 4 + index,
                "n_categorical_features": 0,
                "numeric_feature_ratio": 1.0,
                "categorical_feature_ratio": 0.0,
                "missing_ratio": 0.0,
                "duplicate_ratio": 0.0,
                "target_cardinality": 2,
                "feature_sample_ratio": 0.08,
                "dataset_dimensionality": (50 + index * 20) * (4 + index),
                "average_numeric_feature_std": 1.0,
                "average_numeric_feature_mean": 0.0,
                "average_numeric_feature_missing_ratio": 0.0,
                "class_imbalance_ratio": 1.0,
                "problem_type": "classification",
                "selected_population_size": population,
                "selected_generations": generations,
            }
        )
    model_path = tmp_path / "meta_model.joblib"
    train_meta_model(pd.DataFrame(rows), model_path)

    prediction = predict_configuration(rows[0], model_path, fallback={"population_size": 10, "generations": 2})

    assert prediction["population_size"] == 3
    assert prediction["generations"] == 1


def test_configuration_selection_prioritizes_validation_f1():
    trials = [
        {"success": True, "f1_score": 0.8, "accuracy": 0.95, "runtime_seconds": 1, "population_size": 3, "generations": 1},
        {"success": True, "f1_score": 0.9, "accuracy": 0.9, "runtime_seconds": 2, "population_size": 5, "generations": 2},
    ]

    selected = choose_best_trial(trials, "classification")

    assert selected["population_size"] == 5


def test_configuration_selection_accepts_validation_metric_names():
    trials = [
        {"success": True, "validation_f1": 0.8, "validation_accuracy": 0.95, "runtime_seconds": 1, "population_size": 3, "generations": 1},
        {"success": True, "validation_f1": 0.9, "validation_accuracy": 0.9, "runtime_seconds": 2, "population_size": 5, "generations": 2},
    ]

    selected = choose_best_trial(trials, "classification")

    assert selected["population_size"] == 5


def test_tpot_trial_uses_validation_unless_test_is_explicit(monkeypatch):
    evaluated = []

    class FakeOptimizer:
        def __init__(self, **kwargs):
            pass

        def optimize(self, X, y):
            return "pipeline"

        def get_label_encoder(self):
            return None

    class FakeEvaluator:
        def __init__(self, problem_type):
            pass

        def evaluate(self, model, X, y, label_encoder=None):
            evaluated.append((X, y))
            return {"accuracy": 1.0, "f1_score": 1.0}

    monkeypatch.setattr(configuration_search, "PipelineOptimizer", FakeOptimizer)
    monkeypatch.setattr(configuration_search, "ModelEvaluator", FakeEvaluator)
    splits = {
        "problem_type": "classification",
        "X_train": [[0]],
        "y_train": [0],
        "X_validation": [[1]],
        "y_validation": [1],
        "X_test": [[2]],
        "y_test": [2],
    }

    validation_result = configuration_search.run_tpot(splits)
    test_result = configuration_search.run_tpot(splits, evaluation_split="test")

    assert validation_result["success"] is True
    assert test_result["success"] is True
    assert evaluated == [([[1]], [1]), ([[2]], [2])]


def test_results_summary_keeps_failures_and_computes_paired_differences():
    summary = summarize_results(
        pd.DataFrame(
            [
                {
                    "population_prediction_mae": None,
                    "predicted_population_size": 5,
                    "predicted_generations": 2,
                    "selected_population_size": 3,
                    "selected_generations": 1,
                    "default_success": True,
                    "manual_success": True,
                    "meta_success": True,
                    "default_accuracy": 0.8,
                    "manual_accuracy": 0.82,
                    "meta_accuracy": 0.81,
                    "default_runtime_seconds": 10,
                    "manual_runtime_seconds": 12,
                    "meta_runtime_seconds": 8,
                },
                {
                    "predicted_population_size": 5,
                    "predicted_generations": 2,
                    "selected_population_size": 5,
                    "selected_generations": 2,
                    "default_success": False,
                    "manual_success": True,
                    "meta_success": True,
                    "default_accuracy": None,
                    "manual_accuracy": 0.7,
                    "meta_accuracy": 0.75,
                    "default_runtime_seconds": 1,
                    "manual_runtime_seconds": 10,
                    "meta_runtime_seconds": 9,
                },
            ]
        )
    )

    assert summary["population_prediction_mae"] == 1.0
    assert summary["generation_prediction_mae"] == 0.5
    assert summary["success_rate_default"] == 0.5
    assert summary["meta_minus_manual_accuracy_mean"] == pytest.approx(0.02)


def test_results_summary_handles_regression_without_accuracy_scores():
    summary = summarize_results(
        pd.DataFrame(
            [
                {
                    "selected_population_size": 3,
                    "selected_generations": 1,
                    "predicted_population_size": 5,
                    "predicted_generations": 2,
                    "default_success": True,
                    "manual_success": True,
                    "meta_success": True,
                    "default_accuracy": None,
                    "manual_accuracy": None,
                    "meta_accuracy": None,
                    "default_runtime_seconds": 10,
                    "manual_runtime_seconds": 12,
                    "meta_runtime_seconds": 8,
                }
            ]
        )
    )

    assert "meta_minus_default_accuracy_mean" not in summary
    assert summary["meta_minus_default_runtime_seconds_mean"] == -2.0