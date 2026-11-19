from automl.pipeline_optimizer import PipelineOptimizer
from automl.pipeline_exporter import PipelineExporter

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


def main():

    print("\n========== EXPORTER TEST ==========\n")

    # Load Iris
    iris = load_iris()

    X = iris.data
    y = iris.target

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # TPOT
    optimizer = PipelineOptimizer(
        problem_type="classification",
        generations=1,
        population_size=3,
        max_time_mins=1,
        max_eval_time_mins=0.1,
        random_state=42
    )

    # Train
    best_pipeline = optimizer.optimize(
        X_train,
        y_train
    )

    # Exporter
    exporter = PipelineExporter()

    # Save
    file_path = exporter.save_pipeline(
        best_pipeline,
        "iris_best_pipeline.pkl"
    )

    print("\nPipeline saved successfully!")
    print("Location:", file_path)

    # Load it again
    loaded_pipeline = exporter.load_pipeline(
        "iris_best_pipeline.pkl"
    )

    print("\nPipeline loaded successfully!")

    # Test loaded model
    prediction = loaded_pipeline.predict(
        [[5.1, 3.5, 1.4, 0.2]]
    )

    print("\nPrediction from loaded pipeline:")
    print(prediction)


if __name__ == "__main__":
    main()