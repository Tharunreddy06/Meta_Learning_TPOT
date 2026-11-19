import numpy as np
import pandas as pd


class DatasetMetaFeatureExtractor:
    """Extract dataset-level meta-features for future meta-learning experiments."""

    def __init__(self, dataset, target_column):
        if not isinstance(dataset, pd.DataFrame):
            raise TypeError("dataset must be a pandas DataFrame.")

        if target_column not in dataset.columns:
            raise ValueError(f"Target column '{target_column}' not found in the dataset.")

        self.dataset = dataset.copy()
        self.target_column = target_column

    def _numeric_columns(self):
        features = self.dataset.drop(columns=[self.target_column])
        return features.select_dtypes(include=[np.number]).columns.tolist()

    def _categorical_columns(self):
        return [
            column for column in self.dataset.columns if column != self.target_column and column not in self._numeric_columns()
        ]

    def _target_series(self):
        return self.dataset[self.target_column]

    def _problem_type(self):
        target = self._target_series()

        if pd.api.types.is_numeric_dtype(target):
            unique_values = target.nunique(dropna=True)
            if unique_values <= 20:
                return "classification"
            return "regression"

        return "classification"

    def _missing_ratio(self):
        total_cells = self.dataset.shape[0] * self.dataset.shape[1]
        if total_cells == 0:
            return 0.0
        missing_cells = self.dataset.isnull().sum().sum()
        return float(missing_cells / total_cells)

    def _duplicate_ratio(self):
        duplicate_rows = self.dataset.duplicated().sum()
        if self.dataset.shape[0] == 0:
            return 0.0
        return float(duplicate_rows / self.dataset.shape[0])

    def _class_statistics(self):
        target = self._target_series()

        if pd.api.types.is_numeric_dtype(target):
            distribution = target.value_counts(dropna=False).to_dict()
            if not distribution:
                return {
                    "class_imbalance_ratio": 0.0,
                    "target_distribution": {},
                    "class_distribution_summary": {},
                }

            counts = pd.Series(distribution)
            max_count = counts.max()
            min_count = counts.min()
            imbalance_ratio = float(max_count / min_count) if min_count > 0 else 0.0
            return {
                "class_imbalance_ratio": imbalance_ratio,
                "target_distribution": distribution,
                "class_distribution_summary": {
                    "mean": float(counts.mean()),
                    "std": float(counts.std(ddof=0)) if len(counts) > 1 else 0.0,
                    "max": float(max_count),
                    "min": float(min_count),
                },
            }

        distribution = target.value_counts(dropna=False).to_dict()
        counts = pd.Series(distribution)
        max_count = counts.max() if not counts.empty else 0
        min_count = counts.min() if not counts.empty else 0
        imbalance_ratio = float(max_count / min_count) if min_count > 0 else 0.0

        return {
            "class_imbalance_ratio": imbalance_ratio,
            "target_distribution": distribution,
            "class_distribution_summary": {
                "mean": float(counts.mean()) if not counts.empty else 0.0,
                "std": float(counts.std(ddof=0)) if len(counts) > 1 else 0.0,
                "max": float(max_count),
                "min": float(min_count),
            },
        }

    def _regression_statistics(self):
        target = self._target_series()
        if pd.api.types.is_numeric_dtype(target):
            return {
                "target_mean": float(target.mean()) if not target.empty else 0.0,
                "target_std": float(target.std(ddof=0)) if target.shape[0] > 1 else 0.0,
                "target_min": float(target.min()) if not target.empty else 0.0,
                "target_max": float(target.max()) if not target.empty else 0.0,
                "target_skewness": float(target.skew()) if target.shape[0] > 2 else 0.0,
            }
        return {
            "target_mean": None,
            "target_std": None,
            "target_min": None,
            "target_max": None,
            "target_skewness": None,
        }

    def extract_all(self):
        X = self.dataset.drop(columns=[self.target_column])
        target = self._target_series()
        numeric_columns = self._numeric_columns()
        categorical_columns = self._categorical_columns()
        n_numeric = len(numeric_columns)
        n_categorical = len(categorical_columns)
        problem_type = self._problem_type()

        features = {
            "n_samples": int(self.dataset.shape[0]),
            "n_features": int(X.shape[1]),
            "n_numeric_features": int(n_numeric),
            "n_categorical_features": int(n_categorical),
            "numeric_feature_ratio": float(n_numeric / X.shape[1]) if X.shape[1] > 0 else 0.0,
            "categorical_feature_ratio": float(n_categorical / X.shape[1]) if X.shape[1] > 0 else 0.0,
            "missing_ratio": self._missing_ratio(),
            "duplicate_ratio": self._duplicate_ratio(),
            "target_cardinality": int(target.nunique(dropna=False)),
            "feature_sample_ratio": float(X.shape[1] / self.dataset.shape[0]) if self.dataset.shape[0] > 0 else 0.0,
            "problem_type": problem_type,
            "target_dtype": str(target.dtype),
            "dataset_dimensionality": float(X.shape[0] * X.shape[1]),
            "average_numeric_feature_std": float(
                X[numeric_columns].std(ddof=0).mean()
            ) if numeric_columns else 0.0,
            "average_numeric_feature_mean": float(
                X[numeric_columns].mean().mean()
            ) if numeric_columns else 0.0,
            "average_numeric_feature_missing_ratio": float(
                X[numeric_columns].isnull().mean().mean()
            ) if numeric_columns else 0.0,
            "target_distribution": {},
            "class_distribution": {},
            "class_imbalance_ratio": 0.0,
        }

        class_details = self._class_statistics()
        features["class_imbalance_ratio"] = class_details["class_imbalance_ratio"]
        features["target_distribution"] = class_details["target_distribution"]
        features["class_distribution"] = class_details["target_distribution"]
        features["class_distribution_summary"] = class_details["class_distribution_summary"]

        if problem_type == "regression":
            regression_stats = self._regression_statistics()
            features.update(regression_stats)

        return features
