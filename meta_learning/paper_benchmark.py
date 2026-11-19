"""Run the three benchmark datasets described in the Meta-TPOT paper."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from automl.pipeline_optimizer import PipelineOptimizer
from automl.preprocessor import DataPreprocessor
from meta_learning.common import (
    DEFAULT_MODEL_PATH,
    PROJECT_ROOT,
    canonical_meta_features,
    load_dataset,
)
from meta_learning.train_meta_model import predict_configuration


PAPER_DATASETS = (
    {
        "name": "iris",
        "path": PROJECT_ROOT / "data" / "uploads" / "iris.csv",
        "target": "target",
        "expected_rows": 150,
        "reported_accuracy": 0.9333,
    },
    {
        "name": "telecom_churn",
        "path": PROJECT_ROOT / "data" / "uploads" / "telecom_churn.csv",
        "target": "Churn",
        "expected_rows": 3333,
        "reported_accuracy": 0.8546,
    },
    {
        "name": "breast_cancer",
        "path": PROJECT_ROOT / "data" / "uploads" / "meta_tpot_test_breast_cancer.csv",
        "target": "diagnosis",
        "expected_rows": 569,
        "reported_accuracy": 0.9737,
    },
)

PAPER_REPORTED_BREAST_CANCER_CONFIGURATION = {
    "population_size": 10,
    "generations": 3,
}


def _decode_predictions(predictions, label_encoder):
    values = np.asarray(predictions)
    if label_encoder is None or not values.size or not np.issubdtype(values.dtype, np.number):
        return predictions
    class_codes = np.arange(len(label_encoder.classes_))
    if np.all(np.isin(values, class_codes)):
        return label_encoder.inverse_transform(values.astype(int))
    return predictions


def run_paper_benchmark(
    dataset_spec,
    *,
    optimization_mode="thorough",
    model_path=DEFAULT_MODEL_PATH,
    output_dir=PROJECT_ROOT / "data" / "meta_learning" / "paper_benchmark",
    configuration_source="meta_model",
):
    """Train and evaluate one paper benchmark using its stratified 80:20 split."""
    dataset = load_dataset(dataset_spec["path"])
    if len(dataset) != dataset_spec["expected_rows"]:
        raise ValueError(
            f"{dataset_spec['name']} must contain {dataset_spec['expected_rows']} "
            f"rows for the paper benchmark; found {len(dataset)}."
        )
    target_column = dataset_spec["target"]
    if target_column not in dataset.columns:
        raise ValueError(
            f"Target column '{target_column}' is missing from {dataset_spec['path']}."
        )

    X = dataset.drop(columns=[target_column])
    y = dataset[target_column]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )
    train_data = X_train.copy()
    train_data[target_column] = y_train
    meta_features = canonical_meta_features(train_data, target_column)

    settings = PipelineOptimizer.resolve_optimization_mode(optimization_mode)
    prediction = None
    population_size = None
    generations = None
    if dataset_spec["name"] == "breast_cancer":
        if configuration_source == "paper_reported":
            population_size = PAPER_REPORTED_BREAST_CANCER_CONFIGURATION[
                "population_size"
            ]
            generations = PAPER_REPORTED_BREAST_CANCER_CONFIGURATION["generations"]
        else:
            prediction = predict_configuration(meta_features, model_path)
            population_size = prediction["population_size"]
            generations = prediction["generations"]

    preprocessor = DataPreprocessor(dataset, target_column).build_preprocessor(X_train)
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    optimizer = PipelineOptimizer(
        problem_type="classification",
        population_size=population_size,
        generations=generations,
        random_state=42,
        preprocessing=False,
        **settings,
    )
    pipeline = optimizer.optimize(X_train_processed, y_train)
    label_encoder = optimizer.get_label_encoder()

    from automl.evaluator import ModelEvaluator

    metrics = ModelEvaluator("classification").evaluate(
        pipeline,
        X_test_processed,
        y_test,
        label_encoder=label_encoder,
    )
    predictions = _decode_predictions(pipeline.predict(X_test_processed), label_encoder)
    report = classification_report(
        y_test,
        predictions,
        output_dict=True,
        zero_division=0,
    )
    per_class = {
        label: scores
        for label, scores in report.items()
        if label not in {"accuracy", "macro avg", "weighted avg"}
    }

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir = output_dir / "models"
    artifact = {
        "pipeline": pipeline,
        "preprocessor": preprocessor,
        "label_encoder": label_encoder,
    }
    import joblib

    artifact_path = artifact_dir / f"{dataset_spec['name']}_artifact.joblib"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, artifact_path)

    actual_configuration = {
        "population_size": population_size,
        "generations": generations,
        "source": (
            configuration_source if dataset_spec["name"] == "breast_cancer" else "tpot_default"
        ),
    }
    result = {
        "dataset": dataset_spec["name"],
        "dataset_path": str(dataset_spec["path"]),
        "target": target_column,
        "rows": len(dataset),
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "features": X.shape[1],
        "problem_type": "classification",
        "meta_features_train": meta_features,
        "optimization_mode": optimization_mode,
        "max_time_mins": settings["max_time_mins"],
        "max_eval_time_mins": settings["max_eval_time_mins"],
        "cv": settings["cv"],
        "early_stop": settings["early_stop"],
        "configuration": actual_configuration,
        "meta_model_prediction": prediction,
        "paper_reported_accuracy": dataset_spec["reported_accuracy"],
        "test_accuracy": float(metrics["accuracy"]),
        "test_precision_weighted": float(metrics["precision"]),
        "test_recall_weighted": float(metrics["recall"]),
        "test_f1_weighted": float(metrics["f1_score"]),
        "per_class_metrics": per_class,
        "confusion_matrix": metrics["confusion_matrix"],
        "classification_report": metrics["classification_report"],
        "artifact_path": str(artifact_path),
        "seed": 42,
    }
    result_path = output_dir / f"{dataset_spec['name']}_result.json"
    result["result_path"] = str(result_path)
    result_path.write_text(
        json.dumps(result, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return result


def run_paper_benchmarks(
    *,
    optimization_mode="thorough",
    model_path=DEFAULT_MODEL_PATH,
    output_dir=PROJECT_ROOT / "data" / "meta_learning" / "paper_benchmark",
    configuration_source="meta_model",
):
    """Run all three benchmark datasets and persist a combined JSON and CSV report."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = [
        run_paper_benchmark(
            dataset_spec,
            optimization_mode=optimization_mode,
            model_path=model_path,
            output_dir=output_dir,
            configuration_source=configuration_source,
        )
        for dataset_spec in PAPER_DATASETS
    ]
    (output_dir / "paper_benchmark_results.json").write_text(
        json.dumps(results, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {
                key: value
                for key, value in result.items()
                if key not in {"meta_features_train", "per_class_metrics", "confusion_matrix"}
            }
            for result in results
        ]
    ).to_csv(output_dir / "paper_benchmark_results.csv", index=False)
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Execute the Iris, Telecom Churn, and Breast Cancer paper benchmarks."
    )
    parser.add_argument(
        "--optimization-mode",
        choices=("quick", "balanced", "thorough"),
        default="thorough",
        help="Resource budget. 'thorough' is the paper's Complete-equivalent mode.",
    )
    parser.add_argument(
        "--configuration-source",
        choices=("meta_model", "paper_reported"),
        default="meta_model",
        help=(
            "Use the trained meta-model for Breast Cancer, or force the paper's "
            "reported population=10/generations=3 configuration."
        ),
    )
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "data" / "meta_learning" / "paper_benchmark"),
    )
    args = parser.parse_args()
    results = run_paper_benchmarks(
        optimization_mode=args.optimization_mode,
        model_path=args.model_path,
        output_dir=args.output_dir,
        configuration_source=args.configuration_source,
    )
    for result in results:
        print(
            f"{result['dataset']}: test accuracy={result['test_accuracy']:.4f}; "
            f"paper reported={result['paper_reported_accuracy']:.4f}; "
            f"configuration={result['configuration']}"
        )


if __name__ == "__main__":
    main()
