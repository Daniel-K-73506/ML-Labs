
from pathlib import Path
import hashlib
import json
import platform

import joblib
import numpy as np
import pandas as pd
import yaml

from scipy import sparse
from sklearn import __version__ as sklearn_version
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler


# ============================================================
# LAB 4: DATA PIPELINE AND QUALITY
# Dataset: UCI Adult (OpenML version 2)
# ============================================================

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

NUMERIC_COLUMNS = [
    "age",
    "fnlwgt",
    "education-num",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
]

CATEGORICAL_COLUMNS = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

RANGES = {
    "age": (17, 90),
    "hours-per-week": (1, 99),
    "capital-gain": (0, 99999),
    "capital-loss": (0, 4356),
}


# ============================================================
# 1. DATA INGESTION
# ============================================================

print("\n========== 1. DATA INGESTION ==========")

adult = fetch_openml(
    name="adult",
    version=2,
    as_frame=True,
)

# Keep the original downloaded data unchanged.
raw_df = adult.frame.copy(deep=True)

print("Dataset loaded successfully!")
print("Rows:", len(raw_df))
print("Columns:", len(raw_df.columns))
print("\nFirst 5 rows:")
print(raw_df.head())

assert raw_df.shape == (48842, 15), "Unexpected dataset shape"

# OpenML names the target 'class'; the lab calls it 'income'.
target_column = "class"


# ============================================================
# 2. SCHEMA AND DATA QUALITY VALIDATION
# ============================================================

print("\n========== 2. DATA VALIDATION ==========")

schema_results = {}

for column in NUMERIC_COLUMNS:
    valid = pd.api.types.is_integer_dtype(raw_df[column])
    schema_results[column] = bool(valid)
    print(f"{column}: {'PASS' if valid else 'FAIL'}")

for column in CATEGORICAL_COLUMNS + [target_column]:
    valid = isinstance(raw_df[column].dtype, pd.CategoricalDtype)
    schema_results[column] = bool(valid)
    print(f"{column}: {'PASS' if valid else 'FAIL'}")

range_violations = {}

print("\nRange checks:")

for column, (minimum, maximum) in RANGES.items():
    invalid = (
        (raw_df[column] < minimum)
        | (raw_df[column] > maximum)
    )

    count = int(invalid.sum())
    range_violations[column] = count

    print(f"{column}: {count} violations")

# Treat '?' and actual null values as missing for validation.
missing_values = {}

print("\nMissing values:")

for column in raw_df.columns:
    values = raw_df[column].astype("object")
    missing = values.isna() | values.eq("?").fillna(False)
    count = int(missing.sum())

    missing_values[column] = count

    print(f"{column}: {count}")

duplicate_count = int(raw_df.duplicated().sum())

print("\nExact duplicate rows:", duplicate_count)

print("\nTarget distribution:")
print(raw_df[target_column].value_counts())

# Do not silently remove invalid rows.
validation_report = {
    "rows": int(len(raw_df)),
    "columns": int(len(raw_df.columns)),
    "schema_checks": schema_results,
    "range_violations": range_violations,
    "missing_values": missing_values,
    "exact_duplicates": duplicate_count,
}

with open(
    RESULTS / "validation_report.json",
    "w",
    encoding="utf-8",
) as file:
    json.dump(validation_report, file, indent=4)

if not all(schema_results.values()):
    raise ValueError("Schema validation failed")

if any(range_violations.values()):
    raise ValueError("Range validation failed")


# ============================================================
# 3. STRATIFIED TRAIN / TEST SPLIT
# ============================================================

print("\n========== 3. TRAIN / TEST SPLIT ==========")

X = raw_df.drop(columns=[target_column]).copy()
y = raw_df[target_column].copy()

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    stratify=y,
    random_state=RANDOM_STATE,
)

print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))

# Remove exact duplicate records from training only.
train_data = X_train.copy()
train_data[target_column] = y_train

train_duplicates = int(train_data.duplicated().sum())

train_data = train_data.drop_duplicates()

X_train = train_data.drop(columns=[target_column]).copy()
y_train = train_data[target_column].copy()

print("Training duplicates removed:", train_duplicates)
print("Cleaned training rows:", len(X_train))
print("Testing rows unchanged:", len(X_test))

assert len(X_test) == 14653


# ============================================================
# 4. HANDLE MISSING VALUES
# ============================================================

print("\n========== 4. MISSING VALUE POLICY ==========")

# Convert categorical columns to object and map '?' to NaN.
# Do not fit imputers on the test set.

def prepare_missing_values(frame):
    output = frame.copy()

    for column in CATEGORICAL_COLUMNS:
        output[column] = (
            output[column]
            .astype("object")
            .replace("?", np.nan)
        )

    return output


X_train = prepare_missing_values(X_train)
X_test = prepare_missing_values(X_test)

print("Missing values prepared for train-only imputation.")
print("Numeric strategy: median")
print("Categorical strategy: most frequent")


# ============================================================
# 5. FEATURE ENGINEERING
# ============================================================

print("\n========== 5. FEATURE ENGINEERING ==========")

def add_features(frame):
    output = frame.copy()

    output["capital_net"] = (
        output["capital-gain"] - output["capital-loss"]
    )

    output["hours_per_age"] = (
        output["hours-per-week"] / (output["age"] - 16)
    )

    return output


X_train_engineered = add_features(X_train)
X_test_engineered = add_features(X_test)

ENGINEERED_COLUMNS = [
    "capital_net",
    "hours_per_age",
]

print("Features created:")
print("1. capital_net")
print("2. hours_per_age")


# ============================================================
# 6. LEAKAGE-FREE PREPROCESSING
# ============================================================

print("\n========== 6. PREPROCESSING ==========")

def create_preprocessor(training_features, numeric_columns):
    # Clipping limits are calculated from training data only.
    lower = (
        training_features[numeric_columns]
        .quantile(0.01)
        .to_numpy(dtype=float)
    )

    upper = (
        training_features[numeric_columns]
        .quantile(0.99)
        .to_numpy(dtype=float)
    )

    numeric_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median"),
        ),
        (
            "clipper",
            FunctionTransformer(
                np.clip,
                kw_args={
                    "a_min": lower,
                    "a_max": upper,
                },
                validate=False,
            ),
        ),
        (
            "scaler",
            StandardScaler(),
        ),
    ])

    categorical_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="most_frequent"),
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore",
            ),
        ),
    ])

    preprocessor = ColumnTransformer([
        (
            "numeric",
            numeric_pipeline,
            numeric_columns,
        ),
        (
            "categorical",
            categorical_pipeline,
            CATEGORICAL_COLUMNS,
        ),
    ])

    return preprocessor


def train_and_evaluate(
    name,
    train_features,
    test_features,
    numeric_columns,
):
    preprocessor = create_preprocessor(
        train_features,
        numeric_columns,
    )

    # Only the training partition is used for fitting.
    X_train_processed = preprocessor.fit_transform(
        train_features
    )

    X_test_processed = preprocessor.transform(
        test_features
    )

    model = LogisticRegression(
        max_iter=2000,
        random_state=RANDOM_STATE,
    )

    model.fit(X_train_processed, y_train)

    predictions = model.predict(X_test_processed)

    accuracy = accuracy_score(y_test, predictions)

    f1 = f1_score(
        y_test,
        predictions,
        pos_label=">50K",
    )

    print(f"\n{name}")
    print("Accuracy:", round(accuracy, 4))
    print("F1 Score:", round(f1, 4))

    return {
        "accuracy": float(accuracy),
        "f1_score": float(f1),
    }, preprocessor, model, X_train_processed, X_test_processed


# ============================================================
# 7. LOGISTIC REGRESSION BASELINE
# ============================================================

print("\n========== 7. MODEL COMPARISON ==========")

baseline_metrics, _, _, _, _ = train_and_evaluate(
    "Baseline: without engineered features",
    X_train,
    X_test,
    NUMERIC_COLUMNS,
)

enhanced_numeric_columns = (
    NUMERIC_COLUMNS + ENGINEERED_COLUMNS
)

(
    enhanced_metrics,
    fitted_preprocessor,
    fitted_model,
    processed_train,
    processed_test,
) = train_and_evaluate(
    "Enhanced: with engineered features",
    X_train_engineered,
    X_test_engineered,
    enhanced_numeric_columns,
)

comparison = {
    "baseline": baseline_metrics,
    "engineered_features": enhanced_metrics,
    "accuracy_difference": (
        enhanced_metrics["accuracy"]
        - baseline_metrics["accuracy"]
    ),
    "f1_difference": (
        enhanced_metrics["f1_score"]
        - baseline_metrics["f1_score"]
    ),
}

with open(
    RESULTS / "model_comparison.json",
    "w",
    encoding="utf-8",
) as file:
    json.dump(comparison, file, indent=4)

# Save the fitted, reusable objects.
joblib.dump(
    fitted_preprocessor,
    RESULTS / "preprocessor.joblib",
)

joblib.dump(
    fitted_model,
    RESULTS / "logistic_model.joblib",
)

print("\nFitted preprocessor and model saved.")


# ============================================================
# 8. LABEL QUALITY AUDIT
# ============================================================

print("\n========== 8. LABEL QUALITY AUDIT ==========")

# Identify feature-identical records having different labels.
feature_columns = list(X.columns)

grouped_labels = (
    raw_df
    .groupby(
        feature_columns,
        dropna=False,
        observed=True,
    )[target_column]
    .nunique()
)

conflicting_groups = int((grouped_labels > 1).sum())

print("Conflicting feature groups:", conflicting_groups)

# Inspect 50 reproducibly sampled records.
sample = raw_df.sample(
    n=50,
    random_state=RANDOM_STATE,
).copy()

sample_range_violations = {}

for column, (minimum, maximum) in RANGES.items():
    invalid = (
        (sample[column] < minimum)
        | (sample[column] > maximum)
    )

    sample_range_violations[column] = int(invalid.sum())

# Additional data-dictionary consistency checks.
education_mapping = {
    "Preschool": 1,
    "1st-4th": 2,
    "5th-6th": 3,
    "7th-8th": 4,
    "9th": 5,
    "10th": 6,
    "11th": 7,
    "12th": 8,
    "HS-grad": 9,
    "Some-college": 10,
    "Assoc-voc": 11,
    "Assoc-acdm": 12,
    "Bachelors": 13,
    "Masters": 14,
    "Prof-school": 15,
    "Doctorate": 16,
}

expected_education = sample["education"].map(
    education_mapping
)

education_issues = int(
    (
        expected_education
        != sample["education-num"]
    ).sum()
)

sample_label_issues = int(
    (~sample[target_column].isin(["<=50K", ">50K"])).sum()
)

sample_range_violations[
    "education_consistency"
] = education_issues

sample_range_violations[
    "invalid_income_labels"
] = sample_label_issues

sample.to_csv(
    RESULTS / "quality_sample_50.csv",
    index=False,
)

label_report = {
    "conflicting_feature_groups": conflicting_groups,
    "random_sample_size": 50,
    "sample_checks": sample_range_violations,
    "manual_review_status": "Pending human review",
}

with open(
    RESULTS / "label_quality_report.json",
    "w",
    encoding="utf-8",
) as file:
    json.dump(label_report, file, indent=4)

print("Sample size:", len(sample))
print("Sample checks:", sample_range_violations)
print("Sample saved for manual review.")


# ============================================================
# 9. SAVE PROCESSED DATA AND SHA256
# ============================================================

print("\n========== 9. SHA256 HASH ==========")

# Save the transformed features in a sparse format.
processed_train = sparse.csr_matrix(processed_train)
processed_test = sparse.csr_matrix(processed_test)

sparse.save_npz(
    RESULTS / "processed_train.npz",
    processed_train,
)

sparse.save_npz(
    RESULTS / "processed_test.npz",
    processed_test,
)

# Save labels separately, preserving row order.
np.save(
    RESULTS / "y_train.npy",
    y_train.astype(str).to_numpy(),
)

np.save(
    RESULTS / "y_test.npy",
    y_test.astype(str).to_numpy(),
)

# Hash the processed feature and target files.
processed_files = [
    "processed_train.npz",
    "processed_test.npz",
    "y_train.npy",
    "y_test.npy",
]

file_hashes = {}

for filename in processed_files:
    content = (RESULTS / filename).read_bytes()
    file_hashes[filename] = hashlib.sha256(
        content
    ).hexdigest()

with open(
    RESULTS / "sha256_hashes.json",
    "w",
    encoding="utf-8",
) as file:
    json.dump(file_hashes, file, indent=4)

print("Processed data SHA256 hashes:")
for filename, digest in file_hashes.items():
    print(f"{filename}: {digest}")


# ============================================================
# 10. CONFIGURATION YAML
# ============================================================

print("\n========== 10. CONFIGURATION ==========")

config = {
    "lab": "Lab 4 - Data Pipeline and Quality",
    "dataset": {
        "name": "adult",
        "source": "OpenML / UCI Adult",
        "version": 2,
        "raw_rows": int(len(raw_df)),
        "raw_columns": int(len(raw_df.columns)),
        "target": target_column,
    },
    "pipeline": {
        "random_state": RANDOM_STATE,
        "test_size": 0.30,
        "stratified": True,
        "categorical_imputation": "most_frequent",
        "numeric_imputation": "median",
        "numeric_clipping": [0.01, 0.99],
        "duplicates": "remove exact training duplicates",
        "feature_engineering": ENGINEERED_COLUMNS,
    },
    "model": {
        "type": "LogisticRegression",
        "max_iter": 2000,
        "metrics": ["accuracy", "f1"],
        "positive_class": ">50K",
    },
    "environment": {
        "python": platform.python_version(),
        "scikit_learn": sklearn_version,
        "pandas": pd.__version__,
        "numpy": np.__version__,
    },
}

with open(
    ROOT / "config.yaml",
    "w",
    encoding="utf-8",
) as file:
    yaml.safe_dump(
        config,
        file,
        sort_keys=False,
    )

print("config.yaml saved.")


# ============================================================
# 11. DATA CARD
# ============================================================

print("\n========== 11. DATA CARD ==========")

data_card = f"""# Data Card: UCI Adult Dataset

## Source

- Original dataset: UCI Machine Learning Repository, Adult
- Retrieved using OpenML: adult, version 2
- Data based on 1994 US Census information
- License: Verify the current UCI/OpenML dataset terms before reuse

## Problem Definition

Predict whether annual income exceeds $50,000.

- Inputs: 14 original demographic and employment features
- Target: class (<=50K or >50K)
- Task: Binary classification
- Evaluation: Accuracy and F1 score
- Constraint: Reproducible, leakage-free processing

## Dataset Size

- Original rows: {len(raw_df)}
- Original columns: {len(raw_df.columns)}
- Training rows after duplicate removal: {len(X_train)}
- Testing rows: {len(X_test)}
- Original features: {len(feature_columns)}
- Engineered features: {len(ENGINEERED_COLUMNS)}

## Data Quality

- Exact duplicate records in raw data: {duplicate_count}
- Exact duplicate records removed from training: {train_duplicates}
- Feature groups with conflicting labels: {conflicting_groups}
- Randomly sampled records: 50

Missing values and schema results are recorded in
results/validation_report.json.

## Preprocessing

- Stratified 70/30 split before transformation
- Median imputation for numerical features
- Most-frequent imputation for categorical features
- Train-only 1st/99th percentile clipping
- One-hot encoding for categorical features
- Standard scaling of numerical features
- Exact duplicate removal from training data only

## Derived Features

1. capital_net = capital-gain - capital-loss
2. hours_per_age = hours-per-week / (age - 16)

## Model Comparison

Baseline:
- Accuracy: {baseline_metrics["accuracy"]:.4f}
- F1: {baseline_metrics["f1_score"]:.4f}

With engineered features:
- Accuracy: {enhanced_metrics["accuracy"]:.4f}
- F1: {enhanced_metrics["f1_score"]:.4f}

## Known Biases

The dataset reflects historical US demographic and economic
conditions. It contains sensitive attributes including race
and sex. Income outcomes may reflect structural inequalities.

Historical results may not generalize to present-day populations.

## Privacy Concerns

Although publicly distributed, records contain demographic
and employment-related information. Avoid using the dataset
to identify individuals or make consequential decisions.

## Limitations

The labels represent historical income categories, not
individual ability or merit.

Automated quality checks cannot establish that every income
label is factually correct.

The 50-row sample must also be reviewed manually using
results/quality_sample_50.csv.

## Reproducibility

Run the complete experiment using:

python src/pipeline.py

The processed-data hashes are stored in:
results/sha256_hashes.json
"""

(ROOT / "data_card.md").write_text(
    data_card,
    encoding="utf-8",
)

print("data_card.md saved.")


# ============================================================
# 12. PROBLEM DEFINITION
# ============================================================

problem_definition = """# Problem Definition

## Background

The UCI Adult dataset is a binary classification dataset
derived from historical US Census records. It contains
demographic, educational, occupational and financial
attributes. The goal is to predict whether an individual's
annual income is greater than $50,000.

## Inputs and Output

The model uses 14 original features, including age,
workclass, education, marital status, occupation,
relationship, race, sex, capital gain, capital loss,
weekly working hours, and country of origin.

The output is the income class: <=50K or >50K.

## Objectives

Develop a reproducible data-processing and machine-learning
workflow using Python and scikit-learn. Validate the dataset
before modelling, document quality problems, handle missing
values, identify duplicate observations, and evaluate the
effect of two additional engineered features.

## Success Metrics

Accuracy measures the proportion of correctly classified
observations. F1 score measures the balance between
precision and recall for the >50K income class.

Both metrics are calculated on the untouched test partition.
The baseline and engineered-feature models are compared
using the same split.

## Constraints

The dataset must first be divided using a stratified 70/30
train/test split. Imputation, clipping, encoding, scaling,
and model training must learn only from training data.

Test-set labels must not influence preprocessing or
hyperparameter choices. Exact duplicates may be removed
from training, but the test partition remains unchanged.

The workflow must be reproducible with a fixed random seed,
configuration file, saved fitted transformer, quality
reports, and processed-data SHA256 hashes.

## Ethical Considerations

The dataset contains sensitive demographic attributes and
historical income outcomes. Performance must not be
interpreted as proof of fairness. Potential discrimination,
sampling bias and privacy concerns should be documented.
"""

(ROOT / "problem_definition.md").write_text(
    problem_definition,
    encoding="utf-8",
)

print("problem_definition.md saved.")


# ============================================================
# 13. COMPLETION SUMMARY
# ============================================================

print("\n========================================")
print("LAB 4 COMPLETED SUCCESSFULLY")
print("========================================")

print("Original rows:", len(raw_df))
print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))

print("\nBaseline:")
print(baseline_metrics)

print("\nWith engineered features:")
print(enhanced_metrics)

print("\nGenerated files:")
for path in sorted(RESULTS.iterdir()):
    print("-", path.name)

print("\nDocumentation:")
print("- config.yaml")
print("- data_card.md")
print("- problem_definition.md")

print("\nReady for manual quality review and GitHub submission.")
