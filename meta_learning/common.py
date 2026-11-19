import csv
import json
import math
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from automl.meta_feature_extractor import DatasetMetaFeatureExtractor
from automl.preprocessor import DataPreprocessor


PROJECT_ROOT = Path(__file__).resolve().parent.parent

CONFIGURATION_SPACE = [
    {"population_size": 3, "generations": 1},
    {"population_size": 5, "generations": 1},
    {"population_size": 5, "generations": 2},
    {"population_size": 10, "generations": 2},
    {"population_size": 10, "generations": 3},
]

META_FEATURE_COLUMNS = ["N", "F", "P"]
NUMERIC_META_FEATURES = ["N", "F"]
PROBLEM_TYPE_ENCODING = {"classification": 0, "regression": 1}
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "meta_learning" / "meta_model.joblib"


def resolve_problem_type(target):
    """Return a canonical problem-type label for a dataset target."""
    if pd.api.types.is_numeric_dtype(target):
        unique_values = target.nunique(dropna=True)
        if unique_values <= 20:
            return "classification"
        return "regression"
    return "classification"


def canonical_meta_features(dataset, target_column):
    """Return the paper-aligned dataset descriptor [N, F, P]."""
    if not isinstance(dataset, pd.DataFrame):
        raise TypeError("dataset must be a pandas DataFrame.")
    if target_column not in dataset.columns:
        raise ValueError(f"Target column '{target_column}' not found in the dataset.")
    X = dataset.drop(columns=[target_column])
    target = dataset[target_column]
    return {
        "N": int(dataset.shape[0]),
        "F": int(X.shape[1]),
        "P": resolve_problem_type(target),
    }


def load_dataset(path):
    """Read CSV datasets robustly across delimiters and encodings."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset does not exist: {path}")

    sample = path.read_bytes()[:65536]
    encodings = ["utf-8-sig", "utf-8", "latin-1", "cp1252"]
    for encoding in encodings:
        try:
            text = sample.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise ValueError(f"Dataset encoding is not supported: {path}")

    delimiters = [",", ";", "\t", "|"]
    candidate = [
        delimiter
        for delimiter in delimiters
        if delimiter in text
    ]
    if not candidate:
        raise ValueError("Dataset does not contain a recognizable delimiter or column separator.")

    try:
        df = pd.read_csv(path, sep=None, engine="python", encoding=encodings[0])
    except Exception:
        chosen = candidate[0]
        df = pd.read_csv(path, sep=chosen, engine="python", encoding=encodings[0])

    if df.empty:
        raise ValueError("The uploaded dataset is empty.")
    df.columns = [str(column).strip() for column in df.columns]
    if df.columns.has_duplicates:
        raise ValueError("The uploaded dataset contains duplicate column names after trimming whitespace. Please rename or remove duplicates before uploading.")
    if df.columns.empty:
        raise ValueError("The uploaded dataset has no usable columns.")
    if df.shape[1] == 1:
        raise ValueError("The uploaded dataset could not be parsed into columns. Check the delimiter and file format.")

    return df


def parse_dataset_spec(spec):
    """Parse a CLI dataset argument in the form path=target-column."""
    if "=" not in spec:
        raise ValueError(f"Dataset must be specified as path=target: {spec}")
    path, target = spec.rsplit("=", 1)
    if not path or not target:
        raise ValueError(f"Dataset must be specified as path=target: {spec}")
    return Path(path), target


def dataset_identifier(path):
    """Use the supplied relative path as a stable ID, including for duplicate filenames."""
    return Path(path).as_posix()


def normalize_problem_type(value):
    """Normalize a problem-type value to the canonical paper-aligned labels."""
    if value is None:
        return "classification"
    value = str(value).strip().lower()
    if value in {"classification", "cls", "class"}:
        return "classification"
    if value in {"regression", "reg", "regr"}:
        return "regression"
    if value in {"0", "0.0"}:
        return "classification"
    if value in {"1", "1.0"}:
        return "regression"
    return "classification" if value == "classification" else "regression"


def extract_model_features(features):
    """Return a canonical [N, F, P] descriptor for the meta-model."""
    if isinstance(features, dict):
        return {
            "N": int(features.get("N", features.get("n_samples", 0))),
            "F": int(features.get("F", features.get("n_features", 0))),
            "P": normalize_problem_type(features.get("P", features.get("problem_type", "classification"))),
        }
    return canonical_meta_features(features, "target")


def _stratify_labels(y, problem_type, fraction):
    if problem_type != "classification":
        return None
    counts = y.value_counts(dropna=False)
    n_groups = len(counts)
    n_split = math.ceil(len(y) * fraction)
    n_remaining = len(y) - n_split
    if n_groups > 1 and counts.min() >= 2 and n_split >= n_groups and n_remaining >= n_groups:
        return y
    return None


def prepare_splits(dataset, target_column, random_state=42):
    """Create train/validation/test splits and fit preprocessing on training rows only."""
    if target_column not in dataset.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataset.")
    cleaned = dataset.loc[dataset[target_column].notna()].copy()
    if cleaned.empty:
        raise ValueError("Dataset has no rows with a non-missing target.")

    problem_type = resolve_problem_type(cleaned[target_column])
    X = cleaned.drop(columns=[target_column])
    y = cleaned[target_column]
    stratify = _stratify_labels(y, problem_type, 0.2)
    X_train_valid, X_test, y_train_valid, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=random_state,
        stratify=stratify,
    )
    development_data = X_train_valid.copy()
    development_data[target_column] = y_train_valid
    meta_features = canonical_meta_features(development_data, target_column)

    valid_stratify = _stratify_labels(y_train_valid, problem_type, 0.2)
    X_train, X_validation, y_train, y_validation = train_test_split(
        X_train_valid,
        y_train_valid,
        test_size=0.2,
        random_state=random_state + 1,
        stratify=valid_stratify,
    )

    preprocessor = DataPreprocessor(cleaned, target_column).build_preprocessor(X_train)
    X_train = preprocessor.fit_transform(X_train)
    X_validation = preprocessor.transform(X_validation)
    X_test = preprocessor.transform(X_test)

    return {
        "problem_type": problem_type,
        "meta_features": canonical_meta_features(development_data, target_column),
        "raw_meta_features": meta_features,
        "X_train": X_train,
        "X_validation": X_validation,
        "X_test": X_test,
        "y_train": y_train,
        "y_validation": y_validation,
        "y_test": y_test,
        "preprocessor": preprocessor,
    }


def flatten_meta_features(features):
    """Return CSV-friendly scalar meta-feature columns."""
    return extract_model_features(features)


def serialize_features(features):
    return json.dumps(features, sort_keys=True, default=str)
