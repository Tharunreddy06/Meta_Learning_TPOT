from io import BytesIO

import app as web_app


def test_analysis_page_keeps_existing_modes_and_adds_meta_option(tmp_path):
    web_app.app.config.update(TESTING=True, UPLOAD_FOLDER=str(tmp_path))
    client = web_app.app.test_client()

    response = client.post(
        "/upload",
        data={
            "dataset": (
                BytesIO(b"x,target\n1,a\n2,b\n3,a\n4,b\n5,a\n6,b\n"),
                "small.csv",
            )
        },
        content_type="multipart/form-data",
    )
    assert response.status_code == 200

    response = client.post(
        "/analyze",
        data={"file_path": str(tmp_path / "small.csv"), "target_column": "target"},
    )

    assert response.status_code == 200
    assert b"Use Meta-TPOT configuration" in response.data
    assert b'value="quick"' in response.data
    assert b'value="balanced"' in response.data
    assert b'value="thorough"' in response.data


def test_meta_training_reports_missing_model_artifact(tmp_path, monkeypatch):
    web_app.app.config.update(TESTING=True)
    monkeypatch.setattr(web_app, "DEFAULT_MODEL_PATH", tmp_path / "missing-meta-model.joblib")
    dataset_path = tmp_path / "small.csv"
    dataset_path.write_text("x,target\n1,a\n2,b\n3,a\n4,b\n5,a\n6,b\n", encoding="utf-8")
    client = web_app.app.test_client()

    response = client.post(
        "/train",
        data={
            "file_path": str(dataset_path),
            "target_column": "target",
            "optimization_mode": "quick",
            "use_meta_learning": "true",
        },
        follow_redirects=True,
    )

    assert response.status_code == 200
    assert b"Meta-TPOT model is not trained yet" in response.data
    assert b"missing-meta-model.joblib" not in response.data


def test_training_reports_missing_tpot_dependency(tmp_path, monkeypatch):
    dataset_path = tmp_path / "small.csv"
    dataset_path.write_text("x,target\n1,a\n2,b\n", encoding="utf-8")
    monkeypatch.setattr(web_app, "PipelineOptimizer", None)
    monkeypatch.setattr(web_app, "_PIPELINE_IMPORT_ERROR", ImportError("missing TPOT"))

    response = web_app.app.test_client().post(
        "/train",
        data={"file_path": str(dataset_path), "target_column": "target"},
        follow_redirects=True,
    )

    assert response.status_code == 200
    assert b"TPOT dependencies are not installed" in response.data
    assert b"missing TPOT" not in response.data


def test_meta_prediction_is_passed_to_tpot_and_displayed(tmp_path, monkeypatch):
    import automl.evaluator as evaluator_module
    import automl.pipeline_exporter as exporter_module

    dataset_path = tmp_path / "small.csv"
    dataset_path.write_text(
        "x,target\n" + "\n".join(f"{index},{'a' if index % 2 else 'b'}" for index in range(20)),
        encoding="utf-8",
    )
    prediction = {
        "population_size": 5,
        "generations": 2,
        "training_dataset_count": 4,
    }
    optimizer_parameters = {}
    exported_artifact = {}

    class FakeOptimizer:
        @staticmethod
        def resolve_optimization_mode(mode):
            return {
                "max_time_mins": 1,
                "max_eval_time_mins": 0.1,
                "cv": 3,
                "early_stop": 2,
                "verbose": 0,
            }

        def __init__(self, **kwargs):
            optimizer_parameters.update(kwargs)

        def optimize(self, X, y):
            return object()

        def get_label_encoder(self):
            return None

    class FakeEvaluator:
        def __init__(self, problem_type):
            pass

        def evaluate(self, model, X, y, label_encoder=None):
            return {
                "accuracy": 0.8,
                "precision": 0.8,
                "recall": 0.8,
                "f1_score": 0.8,
                "confusion_matrix": [[2, 0], [0, 2]],
                "classification_report": "test report",
            }

    class FakeExporter:
        def __init__(self, output_directory):
            pass

        def save_pipeline(self, pipeline, filename):
            return filename

        def save_label_encoder(self, label_encoder, filename):
            return filename

        def save_deployment_artifact(self, pipeline, preprocessor, label_encoder=None):
            exported_artifact.update(
                pipeline=pipeline,
                preprocessor=preprocessor,
                label_encoder=label_encoder,
            )
            return "best_model_artifact.pkl"

    monkeypatch.setattr(web_app, "PipelineOptimizer", FakeOptimizer)
    monkeypatch.setattr(web_app, "predict_configuration", lambda features, path: prediction)
    monkeypatch.setattr(evaluator_module, "ModelEvaluator", FakeEvaluator)
    monkeypatch.setattr(exporter_module, "PipelineExporter", FakeExporter)
    web_app.app.config.update(TESTING=True)

    response = web_app.app.test_client().post(
        "/train",
        data={
            "file_path": str(dataset_path),
            "target_column": "target",
            "optimization_mode": "quick",
            "use_meta_learning": "true",
        },
    )

    assert response.status_code == 200
    assert optimizer_parameters["population_size"] == prediction["population_size"]
    assert optimizer_parameters["generations"] == prediction["generations"]
    assert b"<dd>5</dd>" in response.data
    assert b"<dd>2</dd>" in response.data
    assert b"What ran for this dataset" in response.data
    assert b"16 train / 4 test" in response.data
    assert b"N=16" in response.data
    assert b"Population 5" in response.data
    assert b"3-fold CV" in response.data
    assert b"Deployment artifact: best_model_artifact.pkl" in response.data
    assert exported_artifact["pipeline"] is not None
    assert exported_artifact["preprocessor"] is not None