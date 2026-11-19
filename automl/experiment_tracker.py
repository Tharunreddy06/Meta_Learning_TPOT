import json
import os
from datetime import datetime

import pandas as pd


class ExperimentTracker:
    """Store TPOT experiment metadata in a simple CSV-friendly format for later meta-learning use."""

    def __init__(self, output_path="data/meta_dataset.csv"):
        self.output_path = output_path
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    def _serialize_value(self, value):
        if isinstance(value, (dict, list, tuple)):
            return json.dumps(value, sort_keys=True)
        return value

    def log_experiment(self, record):
        if not isinstance(record, dict):
            raise TypeError("record must be a dictionary.")

        row = {
            "timestamp": datetime.utcnow().isoformat(timespec="seconds"),
            "dataset_name": record.get("dataset_name"),
            "problem_type": record.get("problem_type"),
            "n_samples": record.get("n_samples"),
            "n_features": record.get("n_features"),
            "dataset_meta_features": self._serialize_value(record.get("dataset_meta_features", {})),
            "tpot_configuration": self._serialize_value(record.get("tpot_configuration", {})),
            "best_pipeline_score": record.get("best_pipeline_score"),
            "evaluation_score": record.get("evaluation_score"),
            "optimization_runtime_seconds": record.get("optimization_runtime_seconds"),
            "random_state": record.get("random_state"),
            "success": bool(record.get("success", False)),
            "error_message": record.get("error_message"),
        }

        if os.path.exists(self.output_path):
            df = pd.read_csv(self.output_path)
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        else:
            df = pd.DataFrame([row])

        df.to_csv(self.output_path, index=False)
        return row

    def load_records(self):
        if not os.path.exists(self.output_path):
            return pd.DataFrame()
        return pd.read_csv(self.output_path)
