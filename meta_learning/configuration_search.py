import time

from automl.evaluator import ModelEvaluator
from automl.pipeline_optimizer import PipelineOptimizer


def run_tpot(
    splits,
    population_size=None,
    generations=None,
    max_time_mins=1.0,
    max_eval_time_mins=0.1,
    cv=3,
    random_state=42,
    evaluation_split="validation",
):
    """Fit one actual TPOT run and evaluate it on the supplied validation or test split."""
    started = time.perf_counter()
    result = {
        "population_size": population_size,
        "generations": generations,
        "runtime_seconds": None,
        "success": False,
        "error_message": None,
        "accuracy": None,
        "f1_score": None,
        "mae": None,
        "r2_score": None,
        "pipeline": None,
    }
    optimizer = None
    try:
        optimizer = PipelineOptimizer(
            problem_type=splits["problem_type"],
            population_size=population_size,
            generations=generations,
            max_time_mins=max_time_mins,
            max_eval_time_mins=max_eval_time_mins,
            cv=cv,
            early_stop=None,
            random_state=random_state,
            n_jobs=1,
            verbose=0,
        )
        pipeline = optimizer.optimize(splits["X_train"], splits["y_train"])
        if evaluation_split not in {"validation", "test"}:
            raise ValueError("evaluation_split must be 'validation' or 'test'.")
        X_eval = splits[f"X_{evaluation_split}"]
        y_eval = splits[f"y_{evaluation_split}"]
        metrics = ModelEvaluator(splits["problem_type"]).evaluate(
            pipeline,
            X_eval,
            y_eval,
            label_encoder=optimizer.get_label_encoder()
            if splits["problem_type"] == "classification"
            else None,
        )
        result["success"] = True
        result["pipeline"] = pipeline
        if splits["problem_type"] == "classification":
            result["accuracy"] = float(metrics["accuracy"])
            result["f1_score"] = float(metrics["f1_score"])
        else:
            result["mae"] = float(metrics["mae"])
            result["r2_score"] = float(metrics["r2_score"])
    except Exception as exc:
        result["error_message"] = f"{type(exc).__name__}: {exc}"
    finally:
        result["runtime_seconds"] = time.perf_counter() - started
    return result
