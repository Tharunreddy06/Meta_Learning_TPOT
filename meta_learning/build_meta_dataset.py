import argparse
from pathlib import Path

import pandas as pd

from meta_learning.common import (
    CONFIGURATION_SPACE,
    dataset_identifier,
    flatten_meta_features,
    load_dataset,
    parse_dataset_spec,
    prepare_splits,
    serialize_features,
    PROJECT_ROOT,
)
from meta_learning.configuration_search import run_tpot


def choose_best_trial(trials, problem_type):
    """Select by validation quality first, then runtime; never choose by inspection."""
    successful = [trial for trial in trials if trial["success"]]
    if not successful:
        return None
    if problem_type == "classification":
        return max(
            successful,
            key=lambda row: (
                row.get("f1_score", row.get("validation_f1", float("-inf"))),
                row.get("accuracy", row.get("validation_accuracy", float("-inf"))),
                -row["runtime_seconds"],
                -row["population_size"],
                -row["generations"],
            ),
        )
    return max(
        successful,
        key=lambda row: (
            row.get("r2_score", row.get("validation_r2", float("-inf")))
            if row.get("r2_score", row.get("validation_r2")) is not None
            else float("-inf"),
            -(row.get("mae", row.get("validation_mae", float("inf")))
              if row.get("mae", row.get("validation_mae")) is not None
              else float("inf")),
            -row["runtime_seconds"],
            -row["population_size"],
            -row["generations"],
        ),
    )


def build_meta_dataset(
    dataset_specs,
    output_dir=PROJECT_ROOT / "data" / "meta_learning",
    max_time_mins=1.0,
    max_eval_time_mins=0.1,
    cv=3,
    random_state=42,
    synthetic_datasets=None,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    trials_path = output_dir / "configuration_trials.csv"
    features_path = output_dir / "dataset_meta_features.csv"
    meta_dataset_path = output_dir / "meta_dataset.csv"
    trial_rows = pd.read_csv(trials_path).to_dict("records") if trials_path.exists() else []
    feature_rows = pd.read_csv(features_path).to_dict("records") if features_path.exists() else []
    selected_rows = pd.read_csv(meta_dataset_path).to_dict("records") if meta_dataset_path.exists() else []
    synthetic_datasets = {dataset_identifier(path) for path in (synthetic_datasets or [])}

    for spec in dataset_specs:
        path, target = parse_dataset_spec(spec)
        dataset_name = dataset_identifier(path)
        dataset = load_dataset(path)
        splits = prepare_splits(dataset, target, random_state)
        common = {
            "dataset_name": dataset_name,
            "dataset_path": str(path),
            "target_column": target,
            "dataset_provenance": (
                "synthetic" if dataset_name in synthetic_datasets else "unverified"
            ),
            "problem_type": splits["problem_type"],
            **flatten_meta_features(splits["raw_meta_features"]),
        }
        feature_rows = [row for row in feature_rows if row.get("dataset_name") != dataset_name]
        feature_rows.append(common)
        dataset_trials = []

        for configuration in CONFIGURATION_SPACE:
            run_settings = {
                **configuration,
                "max_time_mins": max_time_mins,
                "max_eval_time_mins": max_eval_time_mins,
                "cv": cv,
                "random_state": random_state,
            }
            matching_trials = [
                row
                for row in trial_rows
                if row.get("dataset_name") == dataset_name
                and all(
                    pd.to_numeric(pd.Series([row.get(key)]), errors="coerce").iloc[0]
                    == value
                    for key, value in run_settings.items()
                )
            ]
            successful_trial = next(
                (row for row in reversed(matching_trials) if str(row.get("success")).lower() == "true"),
                None,
            )
            if successful_trial is not None:
                dataset_trials.append(successful_trial)
                print(
                    f"{dataset_name}: reusing successful population={configuration['population_size']} "
                    f"generations={configuration['generations']}"
                )
                continue
            if matching_trials:
                print(
                    f"{dataset_name}: retrying previously failed population={configuration['population_size']} "
                    f"generations={configuration['generations']}"
                )
            outcome = run_tpot(
                splits,
                **configuration,
                max_time_mins=max_time_mins,
                max_eval_time_mins=max_eval_time_mins,
                cv=cv,
                random_state=random_state,
            )
            row = {
                **common,
                **run_settings,
                "attempt": len(matching_trials) + 1,
                "validation_accuracy": outcome["accuracy"],
                "validation_f1": outcome["f1_score"],
                "validation_mae": outcome["mae"],
                "validation_r2": outcome["r2_score"],
                "runtime_seconds": outcome["runtime_seconds"],
                "success": outcome["success"],
                "failure_information": outcome["error_message"],
            }
            trial_rows.append(row)
            dataset_trials.append(row)
            pd.DataFrame(trial_rows).to_csv(trials_path, index=False)
            print(
                f"{dataset_name}: population={configuration['population_size']} "
                f"generations={configuration['generations']} success={outcome['success']}"
            )

        best = choose_best_trial(dataset_trials, splits["problem_type"])
        selected_rows = [row for row in selected_rows if row.get("dataset_name") != dataset_name]
        if best is not None:
            selected_rows.append(
                {
                    **common,
                    "selected_population_size": best["population_size"],
                    "selected_generations": best["generations"],
                    "selected_validation_accuracy": best["validation_accuracy"],
                    "selected_validation_f1": best["validation_f1"],
                    "selected_validation_mae": best["validation_mae"],
                    "selected_validation_r2": best["validation_r2"],
                    "selected_runtime_seconds": best["runtime_seconds"],
                    "successful_configuration_count": sum(
                        trial["success"] for trial in dataset_trials
                    ),
                }
            )
        else:
            print(f"{dataset_name}: no successful configuration; omitted from meta-model labels")

        pd.DataFrame(feature_rows).to_csv(features_path, index=False)
        pd.DataFrame(selected_rows).to_csv(meta_dataset_path, index=False)

    pd.DataFrame(feature_rows).to_csv(features_path, index=False)
    pd.DataFrame(trial_rows).to_csv(trials_path, index=False)
    pd.DataFrame(selected_rows).to_csv(meta_dataset_path, index=False)
    return pd.DataFrame(trial_rows), pd.DataFrame(selected_rows)


def main():
    parser = argparse.ArgumentParser(description="Collect real TPOT trials for meta-learning.")
    parser.add_argument(
        "--dataset",
        action="append",
        required=True,
        help="Dataset path and target, e.g. data/uploads/iris.csv=target; repeat per dataset.",
    )
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "data" / "meta_learning"))
    parser.add_argument("--max-time-mins", type=float, default=1.0)
    parser.add_argument("--max-eval-time-mins", type=float, default=0.1)
    parser.add_argument("--cv", type=int, default=3)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--synthetic-dataset",
        action="append",
        default=[],
        help="Path of a dataset known to be synthetic; repeat as needed.",
    )
    args = parser.parse_args()
    build_meta_dataset(
        args.dataset,
        args.output_dir,
        args.max_time_mins,
        args.max_eval_time_mins,
        args.cv,
        args.random_state,
        args.synthetic_dataset,
    )


if __name__ == "__main__":
    main()
