# Meta-TPOT paper benchmark

The paper benchmark runner executes the three datasets described in the paper:

| Dataset | Project file | Target |
|---|---|---|
| Iris | `data/uploads/iris.csv` | `target` |
| Telecom Churn | `data/uploads/telecom_churn.csv` | `Churn` |
| Breast Cancer Diagnostic | `data/uploads/meta_tpot_test_breast_cancer.csv` | `diagnosis` |

It verifies the reference dataset row counts, makes a stratified 80:20 train/test
split with seed 42, derives `[N, F, P]` from the training rows, fits preprocessing
on training predictors only, and reports weighted and per-class test metrics.
The breast-cancer run uses the trained meta-model by default; use
`--configuration-source paper_reported` to force the paper's reported `(10, 3)`
configuration for a direct configuration comparison.

Run the app's Complete-equivalent search budget:

```powershell
.\venv\Scripts\python.exe -m meta_learning.paper_benchmark --optimization-mode thorough
```

For a shorter smoke benchmark, use `--optimization-mode quick`. Results are
written to `data/meta_learning/paper_benchmark/`, with a JSON report, a
spreadsheet-friendly CSV summary, and one deployment artifact per dataset.
TPOT is stochastic and package/version-dependent, so a valid run may not
reproduce the paper's exact scores or selected pipeline.
