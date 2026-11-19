import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


class ModelEvaluator:

    def __init__(self, problem_type):
        self.problem_type = problem_type

    def _decode_class_predictions(self, predictions, label_encoder):
        """Convert encoded numeric labels back to the original class space when needed."""

        if label_encoder is None:
            return predictions

        try:
            arr = np.asarray(predictions)

            if arr.size == 0:
                return predictions

            if np.issubdtype(arr.dtype, np.number):
                class_codes = np.arange(len(label_encoder.classes_))

                if np.all(np.isin(arr, class_codes)):
                    return label_encoder.inverse_transform(arr)
        except Exception:
            pass

        return predictions

    def _decode_class_targets(self, y_true, label_encoder):
        """Convert encoded target values back to original labels when the task used a label encoder."""

        if label_encoder is None:
            return y_true

        try:
            arr = np.asarray(y_true)

            if np.issubdtype(arr.dtype, np.number):
                class_codes = np.arange(len(label_encoder.classes_))

                if np.all(np.isin(arr, class_codes)):
                    return label_encoder.inverse_transform(arr)
        except Exception:
            pass

        return y_true

    def evaluate_classification(
        self,
        model,
        X_test,
        y_test,
        label_encoder=None
    ):
        """Evaluate a classification model."""

        y_pred = model.predict(X_test)

        if label_encoder is not None:
            y_pred = self._decode_class_predictions(y_pred, label_encoder)
            y_test = self._decode_class_targets(y_test, label_encoder)

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(
                y_test,
                y_pred,
                average="weighted",
                zero_division=0,
            ),
            "recall": recall_score(
                y_test,
                y_pred,
                average="weighted",
                zero_division=0,
            ),
            "f1_score": f1_score(
                y_test,
                y_pred,
                average="weighted",
                zero_division=0,
            ),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
            "classification_report": classification_report(
                y_test,
                y_pred,
                zero_division=0,
            ),
        }

        return metrics

    def evaluate_regression(
        self,
        model,
        X_test,
        y_test
    ):
        """Evaluate a regression model."""

        y_pred = model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)

        metrics = {
            "mae": mean_absolute_error(y_test, y_pred),
            "mse": mse,
            "rmse": np.sqrt(mse),
            "r2_score": r2_score(y_test, y_pred),
        }

        return metrics

    def evaluate(
        self,
        model,
        X_test,
        y_test,
        label_encoder=None
    ):
        """Automatically evaluate based on problem type."""

        if self.problem_type == "classification":
            return self.evaluate_classification(
                model,
                X_test,
                y_test,
                label_encoder,
            )

        elif self.problem_type == "regression":
            return self.evaluate_regression(model, X_test, y_test)

        else:
            raise ValueError(
                "Problem type must be 'classification' or 'regression'."
            )