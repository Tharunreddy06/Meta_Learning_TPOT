from automl.data_analyzer import DataAnalyzer
from sklearn.datasets import load_iris
import pandas as pd


# Load sample Iris dataset
iris = load_iris(as_frame=True)

df = iris.frame

# Convert target numbers to class names
df["target"] = df["target"].map({
    0: "setosa",
    1: "versicolor",
    2: "virginica"
})

# Save temporary CSV
df.to_csv("iris.csv", index=False)


# Create analyzer
analyzer = DataAnalyzer("iris.csv")

# Analyze dataset
result = analyzer.analyze(target_column="target")


print("\n========== DATASET ANALYSIS ==========\n")

print("Rows:", result["basic_info"]["rows"])

print("Columns:", result["basic_info"]["columns"])

print("\nColumn Names:")
print(result["basic_info"]["column_names"])

print("\nNumerical Columns:")
print(result["basic_info"]["numerical_columns"])

print("\nCategorical Columns:")
print(result["basic_info"]["categorical_columns"])

print("\nMissing Values:")
print(result["basic_info"]["missing_values"])

print("\nDuplicate Rows:")
print(result["basic_info"]["duplicate_rows"])

print("\nTarget Information:")
print(result["target"])

print("\nProblem Type:")
print(result["problem_type"])