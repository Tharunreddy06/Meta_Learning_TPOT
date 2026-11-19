import pandas as pd
import numpy as np

from meta_learning.common import load_dataset


class DataAnalyzer:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None

    def load_data(self):
        """Load CSV dataset using the same robust CSV handling as the meta-learning pipeline."""
        self.data = load_dataset(self.file_path)
        return self.data
    def get_basic_info(self):
        """Return basic dataset information."""
        if self.data is None:
            self.load_data()
        numerical_columns = self.data.select_dtypes(
            include=np.number
        ).columns.tolist()
        categorical_columns = self.data.select_dtypes(
            exclude=np.number
        ).columns.tolist()
        missing_values = self.data.isnull().sum()
        missing_values = {
            column: int(count)
            for column, count in missing_values.items()
            if count > 0
        }
        info = {
            "rows": self.data.shape[0],
            "columns": self.data.shape[1],
            "column_names": self.data.columns.tolist(),
            "numerical_columns": numerical_columns,
            "categorical_columns": categorical_columns,
            "missing_values": missing_values,
            "duplicate_rows": int(self.data.duplicated().sum()),
            "data_types": {
                column: str(dtype)
                for column, dtype in self.data.dtypes.items()
            }
        }
        return info
    def get_statistics(self):
        """Return statistical summary for numerical columns."""
        if self.data is None:
            self.load_data()
        statistics = self.data.describe(
            include="all"
        ).transpose()
        return statistics
    def detect_problem_type(self, target_column):
        """
        Detect whether the problem is classification or regression.
        """
        if self.data is None:
            self.load_data()

        if target_column not in self.data.columns:
            raise ValueError(
                f"Target column '{target_column}' not found."
            )

        target = self.data[target_column]

        # Non-numerical target → classification
        if not pd.api.types.is_numeric_dtype(target):
            return "classification"

        # Numerical target
        unique_values = target.nunique()

        # Small number of unique values → classification
        if unique_values <= 20:
            return "classification"

        # Many continuous numerical values → regression
        return "regression"

    def get_target_info(self, target_column):
        """Return information about the target column."""

        if self.data is None:
            self.load_data()

        if target_column not in self.data.columns:
            raise ValueError(
                f"Target column '{target_column}' not found."
            )

        target = self.data[target_column]

        info = {
            "name": target_column,
            "dtype": str(target.dtype),
            "unique_values": int(target.nunique()),
            "missing_values": int(target.isnull().sum())
        }

        # Classification target
        if not pd.api.types.is_numeric_dtype(target):
            info["class_distribution"] = (
                target.value_counts(dropna=False).to_dict()
            )

        return info

    def analyze(self, target_column=None):
        """Run complete dataset analysis."""

        if self.data is None:
            self.load_data()

        result = {
            "basic_info": self.get_basic_info(),
            "statistics": self.get_statistics()
        }

        if target_column:
            result["target"] = self.get_target_info(target_column)
            result["problem_type"] = self.detect_problem_type(
                target_column
            )

        return result