import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from automl.meta_feature_extractor import DatasetMetaFeatureExtractor
from meta_learning.common import (
    META_FEATURE_COLUMNS,
    PROBLEM_TYPE_ENCODING,
    PROJECT_ROOT,
    extract_model_features,
    normalize_problem_type,
)


def _canonical_model_inputs(raw_rows):
    """Return feature rows with the exact paper-aligned [N, F, P] schema."""
    rows = raw_rows.copy()
    aliases = {
        "N": ("N", "n_samples"),
        "F": ("F", "n_features"),
        "P": ("P", "problem_type"),
    }
    for column, candidates in aliases.items():
        source = next((name for name in candidates if name in rows.columns), None)
        if source is None:
            raise ValueError(f"Meta-dataset is missing the '{column}' meta-feature.")
        rows[column] = rows[source]

    rows["N"] = pd.to_numeric(rows["N"], errors="coerce").fillna(0).astype(float)
    rows["F"] = pd.to_numeric(rows["F"], errors="coerce").fillna(0).astype(float)
    rows["P"] = rows["P"].map(
        lambda value: PROBLEM_TYPE_ENCODING[normalize_problem_type(value)]
    ).fillna(PROBLEM_TYPE_ENCODING["classification"]).astype(float)
    return rows[META_FEATURE_COLUMNS]


def train_meta_model(meta_dataset, model_path):
    """Fit population/generation targets using one selected label per dataset."""
    if isinstance(meta_dataset, (str, Path)):
        meta_dataset = pd.read_csv(meta_dataset)
    required = {
        "dataset_name",
        "selected_population_size",
        "selected_generations",
    }
    missing = required.difference(meta_dataset.columns)
    missing_features = [
        column
        for column, aliases in {
            "N": ("N", "n_samples"),
            "F": ("F", "n_features"),
            "P": ("P", "problem_type"),
        }.items()
        if not any(alias in meta_dataset.columns for alias in aliases)
    ]
    missing.update(missing_features)
    if missing:
        raise ValueError(f"Meta-dataset is missing columns: {sorted(missing)}")
    rows = meta_dataset.dropna(
        subset=["selected_population_size", "selected_generations"]
    ).copy()
    if rows["dataset_name"].nunique() < 2:
        raise ValueError("At least two distinct datasets with selected configurations are required.")

    model = RandomForestRegressor(n_estimators=200, random_state=42)
    target = rows[["selected_population_size", "selected_generations"]].astype(float)
    model.fit(
        _canonical_model_inputs(rows),
        target,
    )
    artifact = {
        "model": model,
        "feature_columns": META_FEATURE_COLUMNS,
        "population_generation_candidates": sorted(
            set(
                zip(
                    rows["selected_population_size"].astype(int),
                    rows["selected_generations"].astype(int),
                )
            )
        ),
        "training_dataset_count": int(rows["dataset_name"].nunique()),
    }
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, model_path)
    return artifact


def predict_configuration(features, model_path, fallback=None):
    """Predict numeric settings and snap to the nearest observed valid pair.

    Falls back only when the meta-model cannot be loaded or the prediction genuinely fails.
    A default of 10/2 is not inserted automatically; callers must provide a fallback only if
    they intentionally want one for hard failure cases.
    """
    fallback = fallback or {}
    try:
        artifact = joblib.load(model_path)
        row = pd.DataFrame([
            {
                "N": float(features.get("N", features.get("n_samples", 0))),
                "F": float(features.get("F", features.get("n_features", 0))),
                "P": float(PROBLEM_TYPE_ENCODING[normalize_problem_type(features.get("P", features.get("problem_type", "classification")))]),
            }
        ])
        predicted = artifact["model"].predict(row[META_FEATURE_COLUMNS])[0]
    except Exception:
        if fallback:
            return {
                "population_size": int(fallback.get("population_size", 0)),
                "generations": int(fallback.get("generations", 0)),
                "raw_population_prediction": None,
                "raw_generation_prediction": None,
                "training_dataset_count": 0,
                "used_fallback": True,
            }
        raise

    candidates = artifact["population_generation_candidates"]
    if not candidates:
        if fallback:
            return {
                "population_size": int(fallback.get("population_size", 0)),
                "generations": int(fallback.get("generations", 0)),
                "raw_population_prediction": None,
                "raw_generation_prediction": None,
                "training_dataset_count": artifact.get("training_dataset_count", 0),
                "used_fallback": True,
            }
        raise ValueError("The trained meta-model contains no valid configurations.")
    pop_values = [candidate[0] for candidate in candidates]
    gen_values = [candidate[1] for candidate in candidates]
    population_scale = max(pop_values) - min(pop_values) or 1
    generation_scale = max(gen_values) - min(gen_values) or 1
    selected = min(
        candidates,
        key=lambda pair: (
            ((pair[0] - predicted[0]) / population_scale) ** 2
            + ((pair[1] - predicted[1]) / generation_scale) ** 2,
            pair,
        ),
    )
    return {
        "population_size": int(selected[0]),
        "generations": int(selected[1]),
        "raw_population_prediction": float(predicted[0]),
        "raw_generation_prediction": float(predicted[1]),
        "training_dataset_count": artifact["training_dataset_count"],
        "used_fallback": False,
    }


def debug_meta_model_predictions(dataset_specs, model_path, *, print_debug=True):
    """Compute and print dataset-level meta-feature predictions for debugging."""
    results = []
    for dataset_spec in dataset_specs:
        if "=" not in dataset_spec:
            raise ValueError(f"Dataset must be specified as path=target: {dataset_spec}")
        path_text, target = dataset_spec.rsplit("=", 1)
        dataset_path = Path(path_text)
        dataset_name = dataset_path.name
        dataset = pd.read_csv(dataset_path, sep=None, engine="python")
        features = DatasetMetaFeatureExtractor(dataset, target).extract_all()
        model_features = extract_model_features(features)
        prediction = predict_configuration(model_features, model_path)
        if print_debug:
            print("=== META-LEARNING DEBUG ===")
            print(f"Dataset: {dataset_name}")
            print(f"Meta-features: {model_features}")
            print(f"Predicted population: {prediction['population_size']}")
            print(f"Predicted generations: {prediction['generations']}")
            print("===========================")
        results.append(
            {
                "dataset_name": dataset_name,
                "dataset_path": str(dataset_path),
                "target_column": target,
                "meta_features": model_features,
                "population_size": prediction["population_size"],
                "generations": prediction["generations"],
                "raw_population_prediction": prediction["raw_population_prediction"],
                "raw_generation_prediction": prediction["raw_generation_prediction"],
            }
        )
    return results


def main():
    parser = argparse.ArgumentParser(description="Train the dataset-level TPOT meta-model.")
    parser.add_argument(
        "--meta-dataset",
        default=str(PROJECT_ROOT / "data" / "meta_learning" / "meta_dataset.csv"),
    )
    parser.add_argument("--model-path", default=str(PROJECT_ROOT / "models" / "meta_learning" / "meta_model.joblib"))
    parser.add_argument(
        "--debug-dataset",
        action="append",
        default=None,
        help="Repeat as path=target to print meta-feature predictions for a dataset before TPOT starts.",
    )
    args = parser.parse_args()
    if args.debug_dataset:
        debug_meta_model_predictions(args.debug_dataset, args.model_path)
        return
    artifact = train_meta_model(args.meta_dataset, args.model_path)
    print(
        f"Saved meta-model trained from {artifact['training_dataset_count']} datasets "
        f"to {args.model_path}"
    )


if __name__ == "__main__":
    main()
