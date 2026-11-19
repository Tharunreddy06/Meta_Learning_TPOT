# Dataset-Aware Meta-Learning Extension for TPOT

This project now contains an executable implementation of the proposed workflow. See [meta_learning.md](meta_learning.md) for installation, commands, output files, and interpretation limits.

The implementation builds dataset-level meta-features, evaluates five TPOT candidate configurations using a validation split, trains a supervised model for population size and generations, and evaluates predicted settings with dataset-level leave-one-dataset-out folds. Default, fixed manual, and meta-configured TPOT runs use the same held-out test split and search budget.

Generated CSVs and model artifacts are experimental outputs, not guaranteed results. The current supplied meta-dataset covers four datasets, including two Iris variants; this is too small and potentially correlated to support a broad generalization claim. Run the documented evaluation on diverse datasets and report successes and failures as produced. No research-paper file is edited by this project documentation.

This is a proposed research extension and should be treated as such rather than as a completed or publication-ready system.
