# Lab 4: Data pipeline and quality

## Problem definition

The aim is to predict whether a person's annual income is above $50,000 using the Adult dataset. This is a binary classification problem. Each row has 14 input features, including age, education, workclass, occupation, and working hours. The output is `>50K` or `<=50K`. The model does not receive the income label as an input.

The purpose of this lab is to build a repeatable data pipeline and check data quality before training. The data contains both numbers and categories, so these need different handling. Missing values, extreme values, duplicate records, and possible label problems also need to be checked. Any changes should be explained rather than silently removing rows.

Success is measured using accuracy and F1. Accuracy is the fraction of predictions that are correct. F1 combines precision and recall for the `>50K` class. This matters because higher-income examples are the smaller class, so accuracy alone can hide weak results for that group. Both models use a probability threshold of 0.5.

The main constraints are avoiding leakage and making the experiment reproducible. The split must happen before cleaning, and anything learned from data must use training rows only. Test rows must stay separate and must not be dropped. The configuration, split, fitted preprocessing steps, model results, and processed-data hash are saved so the run can be checked and repeated. This is a learning experiment, not a model ready for decisions about people.

## Data and validation

The data was loaded with `fetch_openml(name='adult', version=2, as_frame=True)`. It had **48,842 rows and 15 columns**, including the label. The source frame was kept unchanged. OpenML calls the target `class`; only the working copy renames it to `income`.

The schema in `config.yaml` defines integer and categorical columns, allowed categories, and the four required numeric ranges. All rows passed the type, category, range, and missing-label checks. Missing inputs were allowed and counted:

| Column | Missing values |
|---|---:|
| workclass | 2,799 |
| occupation | 2,809 |
| native-country | 857 |

Other columns had no missing values. OpenML already turns missing markers into nulls. The pipeline also accepts literal `?` as missing. The full checks are in `results/validation_report.md` and `results/schema_checks.csv`. No row was removed for having a missing input.

## Split and preprocessing

The raw data was split before cleaning, using seed 42. A stratified group split kept identical input rows together, including rows with conflicting labels. Three of ten folds became the test set. This gave **34,188 training rows and 14,654 test rows**, or 30.003% test data. The small difference from exactly 30% keeps duplicate groups together. The positive-class proportions were 23.82% in the final training set and 24.20% in test.

There were no identical raw feature groups shared between training and test. Only exact full-row duplicates in training were removed: **35 rows**. This left **34,153 training rows**. All test rows were kept. Removed source-row IDs and split assignments are saved in `results/`.

The preprocessing steps were:

1. Replace missing numeric values with the training median and missing categories with the training mode.
2. Clip each numeric column to its training 1st and 99th percentiles.
3. Add the derived features for the second experiment.
4. Scale numeric columns and one-hot encode categories. Fit both steps on training data only. Ignore unseen categories when transforming test data.

Test rows used those same fitted transformations. They were not used to calculate medians, modes, percentiles, scaling values, or category lists. Their original values and row count remained unchanged.

For example, the training clipping bounds were age 17-74, working hours 8-80, capital gain 0-15,024, and capital loss 0-2,001. All learned values are saved in `results/train_numeric_statistics.csv` and `results/train_categorical_modes.csv`. Clipping every numeric column follows the lab instruction, but can remove useful extreme values. It can also change `education-num` without changing its original education category. Validation therefore checks the original values before clipping.

The fitted preprocessing pipeline with derived features is saved as `results/preprocessor.joblib`. The baseline version is saved separately.

## Derived features and model results

Two features were added after imputation and clipping:

```text
capital_net = capital-gain - capital-loss
hours_per_age = hours-per-week / (age - 16)
```

Age is at least 17, so the denominator is positive. Both features use input values only and do not use the income label.

Both experiments used LogisticRegression with `C=1`, `solver='lbfgs'`, and `max_iter=3000`. They used the same training and test rows. F1 treats `>50K` as the positive class.

| Experiment | Input features | Encoded features | Test accuracy | Test F1 |
|---|---:|---:|---:|---:|
| Baseline | 14 | 105 | 85.58% | 0.6709 |
| With derived features | 16 | 107 | 85.86% | 0.6791 |

Adding the features improved accuracy by about **0.28 percentage points** and F1 by **0.0081**. This was a small improvement on this split. It does not prove that the features will improve every run or another dataset. Both models converged without a convergence warning.

## Duplicate and label audit

The full raw dataset had 52 extra copies of exact rows. Comparing only the 14 input features gave 57 extra copies. These counts refer to extra rows after keeping the first copy, not the number of duplicate groups.

There were **five feature groups with conflicting income labels**, containing **ten rows**. They are listed in `results/label_conflicts.csv`. Equal recorded inputs do not prove that two rows describe the same person, so this flags possible label problems rather than proving which label is wrong. The labels were not changed. Each group stayed in one split.

The separate manual review is **pending**. A seeded sample of 50 original rows is saved in `results/manual_review_50.csv`, with automatic quality flags. No manual review findings have been recorded.

## Hash and reproducibility

The cleaned dataset with both derived features is saved locally as `data/lab04/processed.csv`. Its SHA256 is:

```text
b796851d8d3e49cf2cf1318fb5c0b71543bc58fc00f98fe2cc71894dcded1ec2
```

The hash covers values, labels, source-row IDs, split names, and row order. CSV output uses fixed float formatting, UTF-8, and LF line endings. The hash and configuration are stored together in `results/run_config.yaml`. Separate hashes also cover the actual scaled and encoded model inputs. The processed data stays in the Git-ignored `data/` folder and can be rebuilt.

From the project root, one command runs the pipeline:

```powershell
.\.venv\Scripts\python.exe -u lab04\src\run_lab.py > lab04\run.log 2>&1
```

The first run needs internet access for OpenML. Later runs use its local cache. The environment uses Python 3.11.8 and the existing pinned packages. NumPy/BLAS uses one thread. Saved models and preprocessing objects were reloaded and reproduced their test predictions. Checks also confirmed training-only medians and clipping bounds, finite model inputs, and unchanged source/test data. Results are in `results/verification.json`.

Two complete runs produced the same processed CSV hash, model-input hashes, and accuracy/F1 results. Four focused tests also passed: detecting invalid values, accepting `?` as missing, handling unseen categories and missing/extreme test values without changing the fitted preprocessor, and rejecting a zero feature denominator. The repeat-run summary is in `results/reproducibility_check.json`.

Re-running replaces generated results. Recorded review fields are preserved for the same sampled rows, but the generated manual-review status returns to pending after a rerun.

## Three conclusions

1. Splitting first and fitting preprocessing on training data only avoids learning cleaning values from test data. Keeping duplicate feature groups together also avoids overlap between splits.
2. The two derived features gave a small improvement in accuracy and F1. F1 was still much lower than accuracy, so the smaller positive class remains harder to predict well.
3. Passing schema checks does not mean the dataset is perfect. Missing values, conflicting labels, and possible bias remain limitations. The separate 50-row manual review is pending.

