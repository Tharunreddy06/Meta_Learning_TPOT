import pandas as pd

from sklearn.datasets import load_iris

from automl.benchmark_runner import BenchmarkRunner


def test_benchmark_runner_returns_structured_results():
    iris = load_iris(as_frame=True)
    df = iris.frame.copy()
    df["target"] = iris.target_names[iris.target]

    runner = BenchmarkRunner()
    result = runner.run_experiment(
        dataset=df,
        target_column="target",
        problem_type="classification",
        tpot_configuration={"max_time_mins": 0.01, "max_eval_time_mins": 0.01, "cv": 2, "early_stop": 1},
        random_state=42,
    )

    assert isinstance(result, dict)
    assert "dataset_name" in result
    assert "problem_type" in result
    assert "dataset_meta_features" in result
    assert "tpot_configuration" in result
    assert "runtime_seconds" in result
    assert "success" in result
    assert "n_samples" in result
    assert "n_features" in result
    assert "timestamp" in result
    assert isinstance(result["dataset_meta_features"], dict)
    assert isinstance(result["tpot_configuration"], dict)
    assert result["n_samples"] == len(df)
    assert result["n_features"] == df.shape[1] - 1
    assert 0 <= result["runtime_seconds"]


def test_benchmark_runner_accepts_simple_dataframe():
    df = pd.DataFrame(
        {
            "feature_1": [0.1, 0.2, 0.3, 0.4],
            "feature_2": [1.0, 2.0, 3.0, 4.0],
            "target": ["a", "a", "b", "b"],
        }
    )

    runner = BenchmarkRunner()
    result = runner.run_experiment(
        dataset=df,
        target_column="target",
        problem_type="classification",
        tpot_configuration={"max_time_mins": 0.01, "max_eval_time_mins": 0.01, "cv": 2},
        random_state=7,
    )

    assert result["problem_type"] == "classification"
    assert result["success"] in {True, False}
    assert result["dataset_name"].startswith("dataset")
