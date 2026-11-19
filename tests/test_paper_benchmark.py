from pathlib import Path

import pandas as pd
import pytest
import joblib

from meta_learning import paper_benchmark


def test_paper_benchmark_uses_stratified_80_20_split_and_training_features(
    tmp_path,
    monkeypatch,
):
    dataset_path = tmp_path / "iris.csv"
    pd.DataFrame(
        {
            "feature": range(150),
            "target": ["a", "b", "c"] * 50,
        }
    ).to_csv(dataset_path, index=False)
    spec = {
        "name": "iris",
        "path": dataset_path,
        "target": "target",
        "expected_rows": 150,
        "reported_accuracy": 0.9333,
    }
    captured = {}

    class FakeOptimizer:
        @staticmethod
        def resolve_optimization_mode(mode):
            return {
                "max_time_mins": 1,
                "max_eval_time_mins": 0.1,
                "cv": 3,
                "early_stop": 2,
                "verbose": 0,
            }

        def __init__(self, **kwargs):
            captured["optimizer_settings"] = kwargs

        def optimize(self, X, y):
            captured["train_rows"] = len(y)
            return FakePipeline()

        def get_label_encoder(self):
            return None

    class FakePipeline:
        @staticmethod
        def predict(X):
            return ["a", "b", "c"] * 10

    monkeypatch.setattr(paper_benchmark, "PipelineOptimizer", FakeOptimizer)
    monkeypatch.setattr(joblib, "dump", lambda artifact, path: Path(path).touch())
    result = paper_benchmark.run_paper_benchmark(
        spec,
        optimization_mode="quick",
        output_dir=tmp_path / "results",
    )

    assert result["train_rows"] == 120
    assert result["test_rows"] == 30
    assert result["meta_features_train"] == {
        "N": 120,
        "F": 1,
        "P": "classification",
    }
    assert captured["train_rows"] == 120
    assert Path(result["artifact_path"]).exists()
    assert Path(result["result_path"]).exists()


def test_paper_benchmark_rejects_non_reference_dataset_size(tmp_path):
    dataset_path = tmp_path / "not_iris.csv"
    pd.DataFrame({"feature": range(10), "target": ["a", "b"] * 5}).to_csv(
        dataset_path,
        index=False,
    )
    spec = {
        "name": "iris",
        "path": dataset_path,
        "target": "target",
        "expected_rows": 150,
        "reported_accuracy": 0.9333,
    }

    with pytest.raises(ValueError, match="must contain 150 rows"):
        paper_benchmark.run_paper_benchmark(spec, output_dir=tmp_path / "results")
