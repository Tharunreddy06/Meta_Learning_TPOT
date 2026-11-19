import numpy as np
import pandas as pd


class Predictor:

    def __init__(self, model):
        if isinstance(model, dict) and {
            "pipeline",
            "preprocessor",
        }.issubset(model):
            self.model = model["pipeline"]
            self.preprocessor = model["preprocessor"]
            self.label_encoder = model.get("label_encoder")
        else:
            self.model = model
            self.preprocessor = None
            self.label_encoder = None

    def predict(self, data):
        """
        Make predictions using the trained pipeline.

        Parameters
        ----------
        data : pandas.DataFrame or dictionary

        Returns
        -------
        predictions : array
        """

        # Convert dictionary to DataFrame
        if isinstance(data, dict):
            data = pd.DataFrame([data])

        if self.preprocessor is not None:
            data = self.preprocessor.transform(data)

        predictions = self.model.predict(data)
        if self.label_encoder is not None:
            values = np.asarray(predictions)
            if values.size and np.issubdtype(values.dtype, np.number):
                class_codes = np.arange(len(self.label_encoder.classes_))
                if np.all(np.isin(values, class_codes)):
                    predictions = self.label_encoder.inverse_transform(
                        values.astype(int)
                    )

        return predictions

    def predict_single(self, data):
        """
        Make a prediction for a single sample.
        """

        predictions = self.predict(data)

        return predictions[0]