import os
from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    flash
)
import pandas as pd
try:
    from automl.pipeline_optimizer import PipelineOptimizer
except ImportError as error:
    PipelineOptimizer = None
    _PIPELINE_IMPORT_ERROR = error
else:
    _PIPELINE_IMPORT_ERROR = None
from automl.meta_feature_extractor import DatasetMetaFeatureExtractor
from meta_learning.common import (
    DEFAULT_MODEL_PATH,
    PROJECT_ROOT,
    canonical_meta_features,
    extract_model_features,
    load_dataset,
)
from meta_learning.train_meta_model import predict_configuration
def extract_best_algorithm(pipeline):
    """Return the actual final estimator class name from a fitted TPOT pipeline."""
    if pipeline is None:
        return "Unknown"
    current = pipeline
    visited = set()

    while hasattr(current, "steps") and current.steps:
        current_id = id(current)

        if current_id in visited:
            break

        visited.add(current_id)
        last_step = current.steps[-1][1]

        if hasattr(last_step, "steps") and last_step.steps:
            current = last_step
            continue

        return last_step.__class__.__name__

    return current.__class__.__name__


app = Flask(__name__)

app.secret_key = "tpot-automl-secret-key"

UPLOAD_FOLDER = str(PROJECT_ROOT / "data" / "uploads")

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

os.makedirs(
    UPLOAD_FOLDER,
    exist_ok=True
)


# =========================================================
# HOME
# =========================================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# =========================================================
# UPLOAD DATASET
# =========================================================

@app.route(
    "/upload",
    methods=["POST"]
)
def upload_dataset():

    if "dataset" not in request.files:

        flash(
            "No dataset selected."
        )

        return redirect(
            url_for("home")
        )


    file = request.files["dataset"]


    if file.filename == "":

        flash(
            "Please select a CSV file."
        )

        return redirect(
            url_for("home")
        )


    if not file.filename.lower().endswith(".csv"):

        flash(
            "Only CSV files are supported."
        )

        return redirect(
            url_for("home")
        )


    # Save file

    file_path = os.path.join(
        app.config["UPLOAD_FOLDER"],
        file.filename
    )

    file.save(
        file_path
    )


    # Read dataset

    try:
        from meta_learning.common import load_dataset

        df = load_dataset(file_path)

    except Exception as e:

        flash(
            f"Could not read dataset: {e}"
        )

        return redirect(
            url_for("home")
        )


    # Dataset information

    dataset_info = {

        "filename": file.filename,

        "rows": df.shape[0],

        "columns": df.shape[1],

        "column_names":
            df.columns.tolist()

    }


    # IMPORTANT:
    # At this point we don't know the target yet.
    # Therefore we use select_target.html.

    return render_template(
        "select_target.html",

        dataset_info=dataset_info,

        file_path=file_path
    )


# =========================================================
# ANALYZE DATASET
# =========================================================

@app.route(
    "/analyze",
    methods=["POST"]
)
def analyze_dataset():

    file_path = request.form.get(
        "file_path"
    )

    target_column = request.form.get(
        "target_column"
    )


    if not file_path:

        flash(
            "Dataset path is missing."
        )

        return redirect(
            url_for("home")
        )


    if not target_column:

        flash(
            "Please select a target column."
        )

        return redirect(
            url_for("home")
        )


    try:

        # Data Analyzer

        from automl.data_analyzer import (
            DataAnalyzer
        )


        analyzer = DataAnalyzer(
            file_path
        )


        result = analyzer.analyze(
            target_column=target_column
        )


        return render_template(

            "analysis.html",

            result=result,

            target_column=target_column,

            filename=os.path.basename(
                file_path
            ),

            file_path=file_path

        )


    except Exception as e:

        import traceback

        traceback.print_exc()

        flash(
            f"Analysis failed: {e}"
        )

        return redirect(
            url_for("home")
        )


# =========================================================
# TRAIN AUTO ML MODEL
# =========================================================

@app.route(
    "/train",
    methods=["POST"]
)
def train_model():

    file_path = request.form.get(
        "file_path"
    )

    target_column = request.form.get(
        "target_column"
    )

    optimization_mode = request.form.get(
        "optimization_mode",
        "quick"
    )
    use_meta_learning = request.form.get("use_meta_learning") == "true"
    meta_prediction = None


    if not file_path or not target_column:

        flash(
            "Dataset or target column is missing."
        )

        return redirect(
            url_for("home")
        )

    if PipelineOptimizer is None:
        app.logger.error("TPOT dependencies are unavailable: %s", _PIPELINE_IMPORT_ERROR)
        flash("TPOT dependencies are not installed. Install them with the project's requirements.txt.")
        return redirect(url_for("home"))


    try:

        # =================================================
        # 1. LOAD DATASET
        # =================================================

        df = load_dataset(file_path)


        if target_column not in df.columns:

            raise ValueError(
                f"Target column '{target_column}' "
                "does not exist."
            )


        # =================================================
        # 2. PREPARE FEATURES AND TARGET
        # =================================================

        from automl.preprocessor import DataPreprocessor

        preprocessor = DataPreprocessor(
            df,
            target_column
        )

        X, y = preprocessor.separate_features_target()


        # =================================================
        # 3. DETECT PROBLEM TYPE
        # =================================================

        from automl.data_analyzer import (
            DataAnalyzer
        )


        analyzer = DataAnalyzer(
            file_path
        )


        problem_type = analyzer.detect_problem_type(
            target_column
        )

        # =================================================
        # 4. TRAIN / TEST SPLIT
        # =================================================

        from sklearn.model_selection import (
            train_test_split
        )


        if problem_type == "classification":

            X_train, X_test, y_train, y_test = (
                train_test_split(
                    X,
                    y,
                    test_size=0.2,
                    random_state=42,
                    stratify=y
                )
            )

        else:

            X_train, X_test, y_train, y_test = (
                train_test_split(
                    X,
                    y,
                    test_size=0.2,
                    random_state=42
                )
            )

        training_data = X_train.copy()
        training_data[target_column] = y_train
        meta_features = canonical_meta_features(training_data, target_column)

        if use_meta_learning:
            raw_meta_features = DatasetMetaFeatureExtractor(
                training_data,
                target_column
            ).extract_all()
            try:
                model_features = extract_model_features(raw_meta_features)
                app.logger.info("=== META-LEARNING DEBUG ===")
                app.logger.info("Dataset: %s", os.path.basename(file_path))
                app.logger.info("Meta-features: %s", model_features)
                meta_prediction = predict_configuration(
                    model_features,
                    DEFAULT_MODEL_PATH,
                )
                app.logger.info("Predicted population: %s", meta_prediction["population_size"])
                app.logger.info("Predicted generations: %s", meta_prediction["generations"])
                app.logger.info("Raw predictions: population=%s, generations=%s", meta_prediction.get("raw_population_prediction"), meta_prediction.get("raw_generation_prediction"))
                app.logger.info("===========================")
            except FileNotFoundError:
                app.logger.exception("Meta-TPOT model artifact is missing")
                flash(
                    "Meta-TPOT model is not trained yet. Please run the meta-learning training process first."
                )
                return redirect(url_for("home"))
            except Exception:
                app.logger.exception("Meta-TPOT configuration prediction failed")
                flash(
                    "Meta-TPOT could not predict a configuration. Check the model and application logs."
                )
                return redirect(url_for("home"))


        # =================================================
        # 5. OPTIMIZATION SETTINGS
        # =================================================

        optimizer_settings = PipelineOptimizer.resolve_optimization_mode(
            optimization_mode
        )

        max_time_mins = optimizer_settings["max_time_mins"]
        max_eval_time_mins = optimizer_settings["max_eval_time_mins"]
        cv = optimizer_settings["cv"]
        early_stop = optimizer_settings["early_stop"]
        verbose = optimizer_settings["verbose"]


        # =================================================
        # 6. TPOT OPTIMIZATION
        # =================================================

        optimizer = PipelineOptimizer(
            problem_type=problem_type,
            population_size=(
                meta_prediction["population_size"] if meta_prediction else None
            ),
            generations=(
                meta_prediction["generations"] if meta_prediction else None
            ),
            max_time_mins=max_time_mins,
            max_eval_time_mins=max_eval_time_mins,
            cv=cv,
            early_stop=early_stop,
            random_state=42,
            verbose=verbose,
            preprocessing=False,
        )

        fitted_preprocessor = preprocessor.build_preprocessor(X_train)
        X_train_processed = fitted_preprocessor.fit_transform(X_train)
        X_test_processed = fitted_preprocessor.transform(X_test)

        best_pipeline = optimizer.optimize(
            X_train_processed,
            y_train
        )


        # =================================================
        # 7. EVALUATE MODEL
        # =================================================

        from automl.evaluator import (
            ModelEvaluator
        )


        evaluator = ModelEvaluator(
            problem_type=problem_type
        )


        # Get TPOT label encoder for classification

        label_encoder = None

        if problem_type == "classification":

            label_encoder = optimizer.get_label_encoder()


        metrics = evaluator.evaluate(
            best_pipeline,
            X_test_processed,
            y_test,
            label_encoder=label_encoder
        )


        # =================================================
        # 8. SAVE PIPELINE
        # =================================================

        from automl.pipeline_exporter import (
            PipelineExporter
        )


        exporter = PipelineExporter(
            output_directory=str(PROJECT_ROOT / "models" / "trained")
        )


        pipeline_path = exporter.save_pipeline(

            best_pipeline,

            "best_pipeline.pkl"

        )


        # =================================================
        # SAVE LABEL ENCODER
        # =================================================

        label_encoder_path = None

        if problem_type == "classification":

            label_encoder = optimizer.get_label_encoder()

            if label_encoder is not None:

                label_encoder_path = (
                    exporter.save_label_encoder(
                        label_encoder,
                        "label_encoder.pkl"
                    )
                )

        deployment_artifact_path = exporter.save_deployment_artifact(
            best_pipeline,
            fitted_preprocessor,
            label_encoder=label_encoder,
        )

        # =================================================
        # 9. DISPLAY PIPELINE
        # =================================================

        pipeline_text = str(
            best_pipeline
        )

        best_algorithm = extract_best_algorithm(
            best_pipeline
        )


        # =================================================
        # 10. RESULTS PAGE
        # =================================================

        return render_template(

            "results.html",

            filename=os.path.basename(
                file_path
            ),

            target_column=target_column,

            problem_type=problem_type,

            optimization_mode=optimization_mode,

            training_rows=len(X_train),

            test_rows=len(X_test),

            feature_count=X.shape[1],

            meta_features=meta_features,

            optimizer_settings=optimizer_settings,

            use_meta_learning=use_meta_learning,

            meta_prediction=meta_prediction,

            metrics=metrics,

            pipeline=pipeline_text,

            best_algorithm=best_algorithm,

            pipeline_path=pipeline_path,

            deployment_artifact_path=deployment_artifact_path,

            label_encoder_path=label_encoder_path

        )


    except Exception:
        app.logger.exception("AutoML training failed")
        flash(
            "AutoML training failed. Check the dataset and target, then review the application logs."
        )

        return redirect(
            url_for("home")
        )


# =========================================================
# RUN FLASK
# =========================================================

if __name__ == "__main__":

    app.run(

        debug=True,

        host="127.0.0.1",

        port=5000

    )