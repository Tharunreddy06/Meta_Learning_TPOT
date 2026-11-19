from automl.pipeline_optimizer import PipelineOptimizer
from automl.predictor import Predictor

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


def main():

    print("\n========== PREDICTOR TEST ==========\n")

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

    # Train TPOT
    optimizer = PipelineOptimizer(
        problem_type="classification",
        generations=1,
        population_size=3,
        max_time_mins=1,
        max_eval_time_mins=0.1,
        random_state=42
    )

    best_pipeline = optimizer.optimize(
        X_train,
        y_train
    )

    # Create predictor
    predictor = Predictor(
        best_pipeline
    )

    # New sample
    new_sample = {
        "feature_0": 5.1,
        "feature_1": 3.5,
        "feature_2": 1.4,
        "feature_3": 0.2
    }

    # Convert dictionary to DataFrame
    import pandas as pd

    new_data = pd.DataFrame(
        [[
            new_sample["feature_0"],
            new_sample["feature_1"],
            new_sample["feature_2"],
            new_sample["feature_3"]
        ]],
        columns=[
            "feature_0",
            "feature_1",
            "feature_2",
            "feature_3"
        ]
    )

    prediction = predictor.predict_single(
        new_data
    )

    print("\nNew Sample:")
    print(new_data)

    print("\nPrediction:")
    print(prediction)


if __name__ == "__main__":
    main()