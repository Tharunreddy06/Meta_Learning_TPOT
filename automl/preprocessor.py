import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class DataPreprocessor:

    def __init__(self, data, target_column):
        self.data = data.copy()
        self.target_column = target_column
        self.preprocessor_ = None

    def separate_features_target(self):
        """Separate input features (X) and target (y)."""

        if self.target_column not in self.data.columns:
            raise ValueError(
                f"Target column '{self.target_column}' not found."
            )

        X = self.data.drop(columns=[self.target_column])
        y = self.data[self.target_column]

        return X, y

    def identify_columns(self, X):
        """Identify numerical and categorical columns."""

        numerical_columns = X.select_dtypes(
            include=["number"]
        ).columns.tolist()

        categorical_columns = [
            column for column in X.columns
            if column not in numerical_columns
        ]

        return numerical_columns, categorical_columns

    def build_preprocessor(self, X):
        """Build a preprocessing pipeline for numeric and categorical features."""

        numerical_columns, categorical_columns = (
            self.identify_columns(X)
        )

        transformers = []

        if numerical_columns:
            numerical_pipeline = Pipeline(
                steps=[
                    (
                        "imputer",
                        SimpleImputer(strategy="median")
                    ),
                    (
                        "scaler",
                        StandardScaler()
                    )
                ]
            )
            transformers.append(
                (
                    "numerical",
                    numerical_pipeline,
                    numerical_columns
                )
            )

        if categorical_columns:
            categorical_pipeline = Pipeline(
                steps=[
                    (
                        "imputer",
                        SimpleImputer(strategy="most_frequent")
                    ),
                    (
                        "encoder",
                        OneHotEncoder(
                            handle_unknown="ignore"
                        )
                    )
                ]
            )
            transformers.append(
                (
                    "categorical",
                    categorical_pipeline,
                    categorical_columns
                )
            )

        if not transformers:
            return Pipeline(
                steps=[
                    ("identity", "passthrough")
                ]
            )

        preprocessor = ColumnTransformer(
            transformers=transformers,
            remainder="drop"
        )

        return preprocessor

    def fit_transform(self, X):
        """Fit preprocessing on features and transform them."""

        preprocessor = self.build_preprocessor(X)
        self.preprocessor_ = preprocessor

        if isinstance(preprocessor, Pipeline):
            return preprocessor.fit_transform(X)

        return preprocessor.fit_transform(X)

    def transform(self, X):
        """Apply the fitted preprocessing pipeline to new data."""

        if self.preprocessor_ is None:
            raise ValueError(
                "Preprocessor has not been fitted yet."
            )

        if isinstance(self.preprocessor_, Pipeline):
            return self.preprocessor_.transform(X)

        return self.preprocessor_.transform(X)

    def get_preprocessing_pipeline(self):
        """Return the preprocessing pipeline and target."""

        X, y = self.separate_features_target()

        preprocessor = self.build_preprocessor(X)

        return X, y, preprocessor