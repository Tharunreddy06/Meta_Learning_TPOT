import numpy as np
import pandas as pd

from automl.pipeline_exporter import PipelineExporter
from automl.predictor import Predictor


class FittedPreprocessor:
    def transform(self, data):
        return data[["value"]].to_numpy()


class PredictionPipeline:
    def predict(self, data):
        return data[:, 0] > 0


class EncodedPredictionPipeline:
    def predict(self, data):
        return data[:, 0].astype(int)


class LabelEncoder:
    classes_ = np.array(["negative", "positive"])

    def inverse_transform(self, values):
        return self.classes_[values]


def test_deployment_artifact_restores_preprocessing_for_raw_data(tmp_path):
    exporter = PipelineExporter(str(tmp_path))
    pipeline = PredictionPipeline()
    preprocessor = FittedPreprocessor()

    path = exporter.save_deployment_artifact(
        pipeline,
        preprocessor,
        label_encoder="classes",
    )
    artifact = exporter.load_deployment_artifact()
    predictor = Predictor(artifact)

    predictions = predictor.predict(pd.DataFrame({"value": [1, -1]}))

    assert path.endswith("best_model_artifact.pkl")
    assert np.array_equal(predictions, [True, False])
    assert artifact["label_encoder"] == "classes"


def test_predictor_decodes_labels_from_deployment_artifact():
    predictor = Predictor(
        {
            "pipeline": EncodedPredictionPipeline(),
            "preprocessor": FittedPreprocessor(),
            "label_encoder": LabelEncoder(),
        }
    )

    predictions = predictor.predict(pd.DataFrame({"value": [0, 1]}))

    assert np.array_equal(predictions, ["negative", "positive"])


def test_predictor_remains_compatible_with_pipeline_only():
    predictor = Predictor(PredictionPipeline())

    predictions = predictor.predict(np.array([[1], [-1]]))

    assert np.array_equal(predictions, [True, False])
