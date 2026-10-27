# Data Card: UCI Adult Dataset

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

- Original rows: 48842
- Original columns: 15
- Training rows after duplicate removal: 34165
- Testing rows: 14653
- Original features: 14
- Engineered features: 2

## Data Quality

- Exact duplicate records in raw data: 52
- Exact duplicate records removed from training: 24
- Feature groups with conflicting labels: 5
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
- Accuracy: 0.8540
- F1: 0.6601

With engineered features:
- Accuracy: 0.8578
- F1: 0.6696

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
