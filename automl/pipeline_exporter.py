import os
import joblib


class PipelineExporter:

    def __init__(self, output_directory="models/trained"):
        self.output_directory = output_directory

        # Create directory if it doesn't exist
        os.makedirs(
            self.output_directory,
            exist_ok=True
        )

    def save_pipeline(
        self,
        pipeline,
        filename="best_pipeline.pkl"
    ):
        """Save trained pipeline to disk."""

        file_path = os.path.join(
            self.output_directory,
            filename
        )

        joblib.dump(
            pipeline,
            file_path
        )

        return file_path

    def load_pipeline(
        self,
        filename="best_pipeline.pkl"
    ):
        """Load a saved pipeline."""

        file_path = os.path.join(
            self.output_directory,
            filename
        )

        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"Pipeline not found: {file_path}"
            )

        pipeline = joblib.load(
            file_path
        )

        return pipeline

    def save_deployment_artifact(
        self,
        pipeline,
        preprocessor,
        label_encoder=None,
        filename="best_model_artifact.pkl",
    ):
        """Save the fitted preprocessing and prediction steps as one reusable artifact."""
        file_path = os.path.join(
            self.output_directory,
            filename
        )

        joblib.dump(
            {
                "pipeline": pipeline,
                "preprocessor": preprocessor,
                "label_encoder": label_encoder,
            },
            file_path
        )

        return file_path

    def load_deployment_artifact(
        self,
        filename="best_model_artifact.pkl"
    ):
        """Load a complete deployment artifact."""
        file_path = os.path.join(
            self.output_directory,
            filename
        )

        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"Deployment artifact not found: {file_path}"
            )

        return joblib.load(file_path)

    def save_label_encoder(
        self,
        label_encoder,
        filename="label_encoder.pkl"
    ):
        """Save TPOT label encoder."""

        file_path = os.path.join(
            self.output_directory,
            filename
        )

        joblib.dump(
            label_encoder,
            file_path
        )

        return file_path