# Meta-Learning Experiment

The `meta_learning` package runs real TPOT trials to learn a mapping from dataset meta-features to `population_size` and `generations`. Existing CSVs and model artifacts are experimental outputs, not assumed or guaranteed results.

## Available datasets

The benchmark classification/regression directories are currently empty. Supplied CSVs include Iris (including two files named `iris.csv` in different folders), student grades, and telecom churn. Dataset target columns can vary and `student-mat.csv` uses semicolon delimiters, so commands require an explicit `path=target-column`. Provenance is not established by the repository; output tables label inputs `unverified` by default. Mark any known synthetic input with the repeated `--synthetic-dataset path` option in both build and evaluation commands.

## Build experiment data

Run from the project root. Repeat `--dataset` for each dataset and use unique paths as shown:

```powershell
.\.venv_fixed\Scripts\python.exe -m pip install -r requirements.txt

.\.venv_fixed\Scripts\python.exe -m meta_learning.build_meta_dataset `
  --dataset data/uploads/iris.csv=target `
  --dataset data/uploads/iris_300_rows.csv=species `
  --dataset data/uploads/student-mat.csv=G3 `
  --dataset data/uploads/telecom_churn.csv=Churn `
  --max-time-mins 1 --max-eval-time-mins 0.1 --cv 3
```

Replace a target name if a supplied file differs. TPOT runs five candidate configurations per dataset. Each run has the same time/evaluation limits, CV, seed, training split, and preprocessing. Meta-features are calculated from the development partition (training plus validation; held-out test excluded), matching the web workflow's training-only features. A validation split ranks successful configurations by weighted F1 then accuracy for classification, or R² then lower MAE for regression; runtime breaks ties. Test rows are not used to choose a configuration or create meta-model inputs. Every trial is checkpointed; successful runs with matching configuration/budget/seed are reused, while prior failed runs remain recorded and are retried.

Outputs under `data/meta_learning/`:

- `dataset_meta_features.csv`: one row per supplied dataset.
- `configuration_trials.csv`: every candidate, validation metrics, runtime, success flag, and failure information.
- `meta_dataset.csv`: one validation-selected configuration per dataset with successful trials.

## Train the model and evaluate LODO

```powershell
.\.venv_fixed\Scripts\python.exe -m meta_learning.train_meta_model
.\.venv_fixed\Scripts\python.exe -m meta_learning.evaluate_meta_tpot `
  --dataset data/uploads/iris.csv=target `
  --dataset data/uploads/iris_300_rows.csv=species `
  --dataset data/uploads/student-mat.csv=G3 `
  --dataset data/uploads/telecom_churn.csv=Churn `
  --max-time-mins 1 --cv 3
```

If provenance confirms a dataset is synthetic, append `--synthetic-dataset data/uploads/iris_300_rows.csv` (or the appropriate path) to both commands; do not classify it from the filename alone.

At least three distinct dataset paths are needed so that each leave-one-dataset-out model has at least two dataset-level training examples. The LODO evaluator fits a fresh model on all other dataset rows for each held-out dataset. It predicts a candidate pair from combinations selected in the training folds, then evaluates default, fixed manual, and meta-configured TPOT on the exact same untouched test split. Search time, evaluation time, CV, preprocessing, and random seed are matched. The default baseline leaves population/generation at the installed TPOT defaults (TPOT 1.1.0: population size 50 and generations unset) while applying the shared wall-clock budget. The manual baseline is the existing explicit test configuration (population size 3, generations 1).

Generated outputs also include `meta_learning_results.csv`, `meta_learning_summary.json`, and comparison plots. Failed methods remain as rows with failure details; plots that compare all three methods only include datasets where each succeeded. Population/generation MAE and exact pair match compare the LODO prediction with the validation-selected label, not test accuracy.

## Web application

Train a deployment model from the generated meta-dataset before selecting Meta-TPOT in the analysis page. The actual fitted scikit-learn pipeline is saved to `models/meta_learning/meta_model.joblib`; training creates the directory, and model paths resolve from the project location rather than the process working directory. The optional checkbox extracts the same training-partition meta-features, predicts an observed population/generation pair, passes both values to `PipelineOptimizer`, and displays them on the results page. If the artifact is missing or cannot load, the app gives a useful message and logs details without showing a traceback.

Start the existing application with:

```powershell
.\.venv_fixed\Scripts\python.exe app.py
```

## Interpretation limits

These supplied files provide only a small number of dataset-level examples, and the two Iris files may be closely related. LODO is the correct leakage-aware evaluation for this setup, but few or correlated datasets cannot establish broad generalization. Report all runs, failures, and metrics as generated; do not describe the data as a representative real-world benchmark without independently verifying its provenance.