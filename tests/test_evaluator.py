from automl.pipeline_optimizer import PipelineOptimizer
from automl.evaluator import ModelEvaluator

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


def main():

    print("\n========== EVALUATOR TEST ==========\n")

    # Load dataset
    iris = load_iris()

    X = iris.data
    y = iris.target

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # TPOT optimizer
    optimizer = PipelineOptimizer(
        problem_type="classification",
        generations=1,
        population_size=3,
        max_time_mins=1,
        max_eval_time_mins=0.1,
        random_state=42
    )

    # Find best pipeline
    best_pipeline = optimizer.optimize(
        X_train,
        y_train
    )

    # Evaluator
    evaluator = ModelEvaluator(
        problem_type="classification"
    )

    # Evaluate
    metrics = evaluator.evaluate(
        best_pipeline,
        X_test,
        y_test
    )

    # Display results
    print("\n========== EVALUATION RESULTS ==========\n")

    print(
        f"Accuracy  : "
        f"{metrics['accuracy']:.4f}"
    )

    print(
        f"Precision : "
        f"{metrics['precision']:.4f}"
    )

    print(
        f"Recall    : "
        f"{metrics['recall']:.4f}"
    )

    print(
        f"F1 Score  : "
        f"{metrics['f1_score']:.4f}"
    )

    print("\nConfusion Matrix:")
    print(metrics["confusion_matrix"])

    print("\nClassification Report:")
    print(metrics["classification_report"])


if __name__ == "__main__":
    main()