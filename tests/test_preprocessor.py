from automl.preprocessor import DataPreprocessor

from sklearn.datasets import load_iris
import pandas as pd


# Load Iris dataset
iris = load_iris(as_frame=True)

df = iris.frame

# Convert target to class names
df["target"] = df["target"].map({
    0: "setosa",
    1: "versicolor",
    2: "virginica"
})


# Create preprocessor
preprocessor = DataPreprocessor(
    data=df,
    target_column="target"
)


# Get X, y and preprocessing pipeline
X, y, pipeline = (
    preprocessor.get_preprocessing_pipeline()
)


print("\n========== PREPROCESSOR TEST ==========\n")

print("Feature columns:")
print(X.columns.tolist())

print("\nTarget:")
print(y.name)

print("\nTarget classes:")
print(y.unique())

print("\nPreprocessing pipeline:")
print(pipeline)