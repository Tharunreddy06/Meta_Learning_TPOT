import pandas as pd

from automl.experiment_tracker import ExperimentTracker


def test_experiment_tracker_creates_records(tmp_path):
    tracker = ExperimentTracker(output_path=str(tmp_path / "meta_dataset.csv"))

    record = {
        "dataset_name": "toy_dataset",
        "problem_type": "classification",
        "n_samples": 50,
        "n_features": 4,
        "dataset_meta_features": {"n_samples": 50, "target_cardinality": 2},
        "tpot_configuration": {"max_time_mins": 1.0, "cv": 3},
        "best_pipeline_score": 0.95,
        "evaluation_score": 0.91,
        "optimization_runtime_seconds": 12.5,
        "random_state": 42,
        "success": True,
        "error_message": None,
    }

    tracker.log_experiment(record)
    rows = tracker.load_records()

    assert len(rows) == 1
    assert rows.iloc[0]["dataset_name"] == "toy_dataset"
    assert rows.iloc[0]["problem_type"] == "classification"
    assert rows.iloc[0]["best_pipeline_score"] == 0.95
    assert bool(rows.iloc[0]["success"]) is True
    assert "dataset_meta_features" in rows.columns
    assert "tpot_configuration" in rows.columns


def test_experiment_tracker_handles_failed_runs(tmp_path):
    tracker = ExperimentTracker(output_path=str(tmp_path / "meta_dataset.csv"))

    tracker.log_experiment(
        {
            "dataset_name": "bad_dataset",
            "problem_type": "classification",
            "n_samples": 10,
            "n_features": 3,
            "dataset_meta_features": {"n_samples": 10},
            "tpot_configuration": {"max_time_mins": 1.0},
            "best_pipeline_score": None,
            "evaluation_score": None,
            "optimization_runtime_seconds": 0.0,
            "random_state": 7,
            "success": False,
            "error_message": "training failed",
        }
    )

    rows = tracker.load_records()
    assert bool(rows.iloc[0]["success"]) is False
    assert rows.iloc[0]["error_message"] == "training failed"
    assert pd.notna(rows.iloc[0]["timestamp"])
