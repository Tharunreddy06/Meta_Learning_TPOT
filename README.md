# Meta-TPOT: Automatic Configuration of TPOT Evolutionary Search Using Dataset-Level Meta-Learning

Meta-TPOT is a meta-learning-based Automated Machine Learning (AutoML) system that predicts suitable TPOT evolutionary search settings for a dataset before the optimization process begins.

The system uses exactly three dataset-level meta-features — **[N, F, P]** — and a Random Forest meta-model to predict the optimal population size (S) and number of generations (G). TPOT then uses these predicted settings to evolve machine-learning pipelines, evaluate the selected model, and export reusable pipelines.

## 1. Problem Statement

TPOT automates machine-learning pipeline optimization using genetic programming. However, users still need to manually configure important search parameters:

- Population size (S)
- Number of generations (G)

Choosing unsuitable values can lead to unnecessary computation or failure to discover better-performing pipelines.

**Meta-TPOT addresses this problem by using dataset-level meta-learning to recommend suitable TPOT search configurations automatically.**

## 2. Project Objectives

- Develop a Flask-based web application for uploading CSV datasets.
- Extract dataset-level meta-features using a strict `[N, F, P]` representation.
- Predict suitable TPOT population size and generations using a Random Forest meta-model.
- Automatically configure and execute TPOT evolutionary search.
- Evaluate the optimized pipeline using relevant performance metrics.
- Export reusable machine-learning pipelines.

## 3. Dataset Representation: [N, F, P]

The meta-learning model uses exactly three features:

| Feature | Description |
|---|---|
| N | Number of observations in the training partition |
| F | Number of input features, excluding the target column |
| P | Problem type: classification or regression |

These features are extracted from the training partition after splitting the dataset.

The same feature definitions, ordering, and encoding are maintained during meta-model training and prediction.

**The predictor uses only these three meta-features.**

## 4. System Workflow

1. **Dataset Upload:** Upload a CSV dataset through the Flask application.
2. **Target Selection:** Select the target column.
3. **Dataset Inspection:** Inspect the dataset structure.
4. **Data Splitting:** Split the dataset into 80% training and 20% testing using random seed 42.
5. **Meta-Feature Extraction:** Extract `[N, F, P]` from the training partition.
6. **Meta-Model Prediction:** Predict suitable population size and generations.
7. **TPOT Configuration:** Configure TPOT using the predicted settings.
8. **Evolutionary Search:** Optimize machine-learning pipelines using genetic programming.
9. **Evaluation and Export:** Evaluate the selected pipeline on held-out test data and export it.

## 5. Methodology

### Dataset-Level Meta-Learning

The meta-model learns from previous dataset-level experiences to predict suitable TPOT search configurations for new datasets.

**Input:** `[N, F, P]`

**Output:** `(S, G)`

### Meta-Learning Model

- Algorithm: Random Forest
- Number of trees: 200
- Input features: Exactly three
- Output: Population size (S) and generations (G)
- Prediction handling: Maps predictions to the nearest valid candidate `(S, G)` pair.

The implementation uses five candidate configuration pairs, ranging from `(3, 1)` to `(10, 3)`.

The application also provides three execution modes:

- Quick
- Balanced
- Thorough

### TPOT Evolutionary Optimization

TPOT uses genetic programming to evolve machine-learning pipelines through:

- Pipeline initialization
- Model evaluation
- Selection
- Crossover
- Mutation

The evolutionary search continues for the configured number of generations and returns the best-performing pipeline found.

## 6. Technology Stack

- Python 3.11
- TPOT
- Flask
- pandas
- scikit-learn
- joblib

## 7. Experimental Results

The project reports the following held-out test results:

| Dataset | TPOT-selected model | Test accuracy |
|---|---|---:|
| Iris | Bagging | 93.33% |
| Telecom Churn | LightGBM | 85.46% |
| Breast Cancer | MLP | 97.37% |

### Additional Breast Cancer Results

- Predicted population size: **10**
- Predicted generations: **3**
- Correct classifications: **111 / 114**
- Malignant-class recall: **92.86%**

These results demonstrate the implemented workflow on the evaluated datasets. They do not yet establish that meta-predicted settings outperform fixed or manually selected TPOT configurations.

## 8. Current Limitations

- The current evaluation reports results on three datasets.
- The meta-model is documented as being trained on four datasets.
- A systematic comparison against fixed and manually selected TPOT configurations is pending.
- Runtime savings and improvements in search quality have not yet been established.

## 9. Future Enhancements

- Compare meta-predicted configurations against manually selected and fixed TPOT settings.
- Include more datasets for meta-model training.
- Evaluate generalization to previously unseen datasets.
- Measure optimization time and computational efficiency.
- Improve reproducibility through consistent data processing and experiment tracking.

## 10. Getting Started

Clone the repository:

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd Meta-TPOT
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the Flask application using the project's actual entry-point file. For example:

```bash
python app.py
```

The entry-point filename may differ depending on the repository structure.

## 11. Team Members

- Thurupu Tharun
- Vemula Yaswanth Kumar
- T Meghana

**Project Guides:** Ms. Lalli K  & Dr. Ashok Sarabu
Assistant Professor, ASAC  
Alliance University, Bengaluru

## 12. Research Reference

**Automatic Configuration of TPOT Evolutionary Search Using Dataset-Level Meta-Learning**

