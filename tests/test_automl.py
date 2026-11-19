from automl.pipeline_optimizer import PipelineOptimizer

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


def main():

    print("\n========== TPOT TEST STARTED ==========\n")

    iris = load_iris()

    X = iris.data
    y = iris.target

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print("Training samples:", len(X_train))
    print("Testing samples:", len(X_test))

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

    accuracy = best_pipeline.score(
        X_test,
        y_test
    )

    print("\n========== FINAL RESULT ==========\n")

    print("Test Accuracy:", accuracy)

    print("\nBest Pipeline:")
    print(best_pipeline)


if __name__ == "__main__":
    main()