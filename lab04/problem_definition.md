# Problem Definition

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
