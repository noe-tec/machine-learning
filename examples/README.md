# Ejemplos: Linear Regression Evaluation & Feature Selection

Standalone Python scripts, each demonstrating one concept using the
[scikit-learn diabetes dataset](https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset)
(442 samples, 10 features: `age`, `sex`, `bmi`, `bp`, `s1`-`s6`; target is a
quantitative measure of diabetes progression one year after baseline). Run
any file directly with `python <file>.py` -- each one is self-contained.

For the closed-form (normal equation) derivation of linear regression, see
`../demos/linear_regression_closed_form.ipynb`.

## Evaluation strategies

How to measure a model's performance correctly.

| File | What it shows |
|---|---|
| `example_evaluation_in_sample.py` | Training and evaluating on the same data (not recommended -- overestimates performance). |
| `example_evaluation_holdout.py` | A single train/test split. |
| `example_evaluation_kfold_manual.py` | K-fold cross-validation, implemented by hand. |
| `example_evaluation_cross_validate.py` | K-fold cross-validation via scikit-learn's `cross_validate`. |
| `example_evaluation_cross_val_predict.py` | K-fold cross-validation via scikit-learn's `cross_val_predict`. |

## Feature selection

Choosing which features to keep, applying the evaluation strategies above.

| File | What it shows |
|---|---|
| `example_feature_selection_sequential_leakage.py` | Sequential selection fit on the full dataset (incorrect -- data leakage). |
| `example_feature_selection_sequential_cv.py` | Sequential selection fit only on the training fold (correct). |
| `example_feature_selection_sequential_optimal.py` | Sweeping the number of features to find the optimum with sequential selection. |
| `example_feature_selection_rfe.py` | Backward selection with RFE (Recursive Feature Elimination). |
| `example_feature_selection_rfe_optimal.py` | Sweeping the number of features to find the optimum with RFE. |
| `example_feature_selection_filter_correlation.py` | Filter-based selection using Pearson correlation. |
