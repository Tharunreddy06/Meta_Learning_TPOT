import time
from datetime import datetime

from automl.experiment_tracker import ExperimentTracker
from automl.meta_feature_extractor import DatasetMetaFeatureExtractor


class BenchmarkRunner:
    """Reusable infrastructure for recording TPOT runs over benchmark datasets."""

    def __init__(self, tracker=None):
        self.tracker = tracker or ExperimentTracker()

    def run_experiment(
        self,
        dataset,
        target_column,
        problem_type,
        tpot_configuration,
        random_state=42,
        dataset_name=None,
    ):
        start_time = time.time()

        if dataset_name is None:
            dataset_name = f"dataset_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

        meta_extractor = DatasetMetaFeatureExtractor(dataset, target_column)
        meta_features = meta_extractor.extract_all()

        result = {
            "dataset_name": dataset_name,
            "problem_type": problem_type,
            "dataset_meta_features": meta_features,
            "tpot_configuration": dict(tpot_configuration or {}),
            "best_pipeline_score": None,
            "evaluation_score": None,
            "optimization_runtime_seconds": 0.0,
            "n_samples": int(dataset.shape[0]),
            "n_features": int(dataset.shape[1] - 1),
            "random_state": random_state,
            "success": False,
            "error_message": None,
            "timestamp": datetime.utcnow().isoformat(timespec="seconds"),
            "runtime_seconds": 0.0,
        }

        try:
            result["runtime_seconds"] = time.time() - start_time
            result["optimization_runtime_seconds"] = result["runtime_seconds"]
            result["success"] = True
            self.tracker.log_experiment(result)
            return result
        except Exception as exc:
            result["error_message"] = str(exc)
            result["success"] = False
            result["runtime_seconds"] = time.time() - start_time
            result["optimization_runtime_seconds"] = result["runtime_seconds"]
            self.tracker.log_experiment(result)
            return result
