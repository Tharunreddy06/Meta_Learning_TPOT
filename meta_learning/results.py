import json
import warnings
from pathlib import Path

import pandas as pd


def summarize_results(results):
    """Calculate transparent prediction errors, paired differences and success rates."""
    frame = results if isinstance(results, pd.DataFrame) else pd.DataFrame(results)
    summary = {
        "held_out_dataset_count": int(len(frame)),
        "population_prediction_mae": None,
        "generation_prediction_mae": None,
        "exact_configuration_match_rate": None,
        "success_rate_default": None,
        "success_rate_manual": None,
        "success_rate_meta": None,
    }
    for method in ("default", "manual", "meta"):
        col = f"{method}_success"
        if col in frame:
            summary[f"success_rate_{method}"] = float(frame[col].fillna(False).astype(bool).mean())

    labels_available = frame[
        ["selected_population_size", "selected_generations"]
    ].notna().all(axis=1)
    prediction_rows = frame[labels_available]
    if not prediction_rows.empty:
        summary["population_prediction_mae"] = float(
            (prediction_rows["predicted_population_size"] - prediction_rows["selected_population_size"])
            .abs()
            .mean()
        )
        summary["generation_prediction_mae"] = float(
            (prediction_rows["predicted_generations"] - prediction_rows["selected_generations"])
            .abs()
            .mean()
        )
        summary["exact_configuration_match_rate"] = float(
            (
                (prediction_rows["predicted_population_size"] == prediction_rows["selected_population_size"])
                & (prediction_rows["predicted_generations"] == prediction_rows["selected_generations"])
            ).mean()
        )

    for baseline in ("default", "manual"):
        paired = frame[
            frame["meta_success"].fillna(False).astype(bool)
            & frame[f"{baseline}_success"].fillna(False).astype(bool)
        ]
        if not paired.empty and "meta_accuracy" in paired:
            baseline_accuracy = f"{baseline}_accuracy"
            if baseline_accuracy in paired:
                accuracy_pairs = paired[["meta_accuracy", baseline_accuracy]].apply(
                    pd.to_numeric,
                    errors="coerce",
                ).dropna()
                if not accuracy_pairs.empty:
                    summary[f"meta_minus_{baseline}_accuracy_mean"] = float(
                        (accuracy_pairs["meta_accuracy"] - accuracy_pairs[baseline_accuracy]).mean()
                    )
        if not paired.empty:
            summary[f"meta_minus_{baseline}_runtime_seconds_mean"] = float(
                (paired["meta_runtime_seconds"] - paired[f"{baseline}_runtime_seconds"]).mean()
            )
            baseline_runtime = paired[f"{baseline}_runtime_seconds"].replace(0, pd.NA)
            summary[f"meta_runtime_reduction_percent_vs_{baseline}_mean"] = float(
                ((baseline_runtime - paired["meta_runtime_seconds"]) / baseline_runtime * 100)
                .dropna()
                .mean()
            ) if baseline_runtime.notna().any() else None
            summary[f"paired_success_count_{baseline}_vs_meta"] = int(len(paired))
    return summary


def write_results(results, output_dir):
    """Write result tables, summary statistics, and plots with failed runs left visible in CSV."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = results if isinstance(results, pd.DataFrame) else pd.DataFrame(results)
    frame.to_csv(output_dir / "meta_learning_results.csv", index=False)
    summary = summarize_results(frame)
    (output_dir / "meta_learning_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8"
    )

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        warnings.warn("Matplotlib is unavailable; CSV and JSON results were written without plots.")
        return summary

    successful = frame[
        frame["default_success"].fillna(False).astype(bool)
        & frame["manual_success"].fillna(False).astype(bool)
        & frame["meta_success"].fillna(False).astype(bool)
    ]
    classification_rows = successful[successful["default_accuracy"].notna()]
    if not classification_rows.empty:
        classification_rows.set_index("dataset_name")[[
            "default_accuracy", "manual_accuracy", "meta_accuracy"
        ]].plot(kind="bar", ylabel="Accuracy", title="Test accuracy (classification; all methods successful)")
        plt.tight_layout()
        plt.savefig(output_dir / "accuracy_comparison.png", dpi=150)
        plt.close()
    if not successful.empty:
        successful.set_index("dataset_name")[[
            "default_runtime_seconds", "manual_runtime_seconds", "meta_runtime_seconds"
        ]].plot(kind="bar", ylabel="Seconds", title="TPOT runtime (datasets with all methods successful)")
        plt.tight_layout()
        plt.savefig(output_dir / "runtime_comparison.png", dpi=150)
        plt.close()

    labeled = frame[
        frame["selected_population_size"].notna()
        & frame["selected_generations"].notna()
    ]
    if not labeled.empty:
        fig, axes = plt.subplots(1, 2, figsize=(9, 4))
        axes[0].scatter(labeled["selected_population_size"], labeled["predicted_population_size"])
        axes[0].set(xlabel="Selected population size", ylabel="Predicted population size")
        axes[1].scatter(labeled["selected_generations"], labeled["predicted_generations"])
        axes[1].set(xlabel="Selected generations", ylabel="Predicted generations")
        fig.tight_layout()
        fig.savefig(output_dir / "configuration_predictions.png", dpi=150)
        plt.close(fig)
    return summary
