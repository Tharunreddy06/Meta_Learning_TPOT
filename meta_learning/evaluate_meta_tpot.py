import argparse
import tempfile
from pathlib import Path

import pandas as pd

from meta_learning.common import (
    dataset_identifier,
    flatten_meta_features,
    load_dataset,
    parse_dataset_spec,
    prepare_splits,
    PROJECT_ROOT,
)
from meta_learning.configuration_search import run_tpot
from meta_learning.results import write_results
from meta_learning.train_meta_model import predict_configuration, train_meta_model


MANUAL_CONFIGURATION = {"population_size": 3, "generations": 1}


def _single_method(splits, config, budget, cv, seed):
    return run_tpot(
        splits,
        population_size=config.get("population_size"),
        generations=config.get("generations"),
        max_time_mins=budget,
        max_eval_time_mins=min(0.1, budget),
        cv=cv,
        random_state=seed,
        evaluation_split="test",
    )


def evaluate_leave_one_dataset_out(
    dataset_specs,
    meta_dataset_path=PROJECT_ROOT / "data" / "meta_learning" / "meta_dataset.csv",
    output_dir=PROJECT_ROOT / "data" / "meta_learning",
    max_time_mins=1.0,
    cv=3,
    random_state=42,
    synthetic_datasets=None,
):
    if len(dataset_specs) < 3:
        raise ValueError("Leave-one-dataset-out evaluation requires at least three datasets.")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    meta_dataset = pd.read_csv(meta_dataset_path)
    synthetic_datasets = {dataset_identifier(path) for path in (synthetic_datasets or [])}
    results = []

    for spec in dataset_specs:
        path, target = parse_dataset_spec(spec)
        dataset_name = dataset_identifier(path)
        heldout = meta_dataset.loc[meta_dataset["dataset_name"] == dataset_name]
        training_rows = meta_dataset.loc[meta_dataset["dataset_name"] != dataset_name]
        if training_rows["dataset_name"].nunique() < 1:
            raise ValueError(f"No other dataset meta-labels available when holding out {dataset_name}.")

        with tempfile.NamedTemporaryFile(
            suffix=".joblib",
            prefix=".loo_meta_model_",
            dir=output_dir,
            delete=False,
        ) as temporary_file:
            temporary_model = Path(temporary_file.name)
        temporary_model.unlink()
        artifact = train_meta_model(training_rows, temporary_model)
        dataset = load_dataset(path)
        splits = prepare_splits(dataset, target, random_state)
        prediction = predict_configuration(splits["meta_features"], temporary_model)
        selected = heldout.iloc[0] if not heldout.empty else None

        methods = {
            "default": {},
            "manual": MANUAL_CONFIGURATION,
            "meta": {
                "population_size": prediction["population_size"],
                "generations": prediction["generations"],
            },
        }
        row = {
            "dataset_name": dataset_name,
            "held_out_dataset": dataset_name,
            "dataset_path": str(path),
            "dataset_provenance": (
                "synthetic"
                if dataset_name in synthetic_datasets
                else (
                    str(selected["dataset_provenance"])
                    if selected is not None and "dataset_provenance" in selected
                    else "unverified"
                )
            ),
            "used_for_meta_model_training": False,
            **flatten_meta_features(splits["raw_meta_features"]),
            "predicted_population_size": prediction["population_size"],
            "predicted_generations": prediction["generations"],
            "meta_model_training_dataset_count": artifact["training_dataset_count"],
            "selected_population_size": (
                int(selected["selected_population_size"]) if selected is not None else None
            ),
            "selected_generations": int(selected["selected_generations"]) if selected is not None else None,
        }
        for method, config in methods.items():
            outcome = _single_method(splits, config, max_time_mins, cv, random_state)
            row[f"{method}_success"] = outcome["success"]
            row[f"{method}_failure_information"] = outcome["error_message"]
            row[f"{method}_runtime_seconds"] = outcome["runtime_seconds"]
            row[f"{method}_accuracy"] = outcome["accuracy"]
            row[f"{method}_f1_score"] = outcome["f1_score"]
            row[f"{method}_mae"] = outcome["mae"]
            row[f"{method}_r2_score"] = outcome["r2_score"]
            print(f"{dataset_name}: {method} success={outcome['success']}")

        row["meta_exact_configuration_match"] = (
            prediction["population_size"] == row["selected_population_size"]
            and prediction["generations"] == row["selected_generations"]
            if row["selected_population_size"] is not None
            else None
        )
        results.append(row)
        temporary_model.unlink(missing_ok=True)

    write_results(results, output_dir)
    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(description="Evaluate Meta-TPOT using dataset-level holdouts.")
    parser.add_argument("--dataset", action="append", required=True, help="Repeat path=target per dataset.")
    parser.add_argument(
        "--meta-dataset",
        default=str(PROJECT_ROOT / "data" / "meta_learning" / "meta_dataset.csv"),
    )
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "data" / "meta_learning"))
    parser.add_argument("--max-time-mins", type=float, default=1.0)
    parser.add_argument("--cv", type=int, default=3)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--synthetic-dataset",
        action="append",
        default=[],
        help="Path of a dataset known to be synthetic; repeat as needed.",
    )
    args = parser.parse_args()
    evaluate_leave_one_dataset_out(
        args.dataset,
        args.meta_dataset,
        args.output_dir,
        args.max_time_mins,
        args.cv,
        args.random_state,
        args.synthetic_dataset,
    )


if __name__ == "__main__":
    main()
