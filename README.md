# Adolescent fMRI impulsivity classification — EFA binary reproduction

## Purpose

Reproduce the original **two-class** random-forest classification of impulsivity using voxelwise fMRI features from **98 adolescents**. This repository contains **no LPA labels, models, or three-class analysis**.

The binary labels are read **unchanged** from `data/adolescents_labels_filtered.csv`. There are **57 class-0** and **41 class-1** participants. The direction (high versus low impulsivity) should be verified against the original label-generation procedure before naming the classes clinically.

## Previously obtained results

| Metric | 30 repeated 80/20 stratified splits |
|---|---:|
| Mean test accuracy | **68.67%** |
| Accuracy standard deviation | **9.64 percentage points** |
| Mean balanced accuracy | **67.57%** |
| Balanced chance reference | **50%** |

The previous per-seed results are supplied in `results/efa_previous_30_seeds.csv`. They were generated in an earlier execution; the packaged script was smoke-tested for seed 900, but all 30 seeds were **not independently rerun at packaging time**. Results are repeated holdouts, not an independent external validation cohort.

## Pipeline

- **Cohort:** 98 adolescents, restricted to `subject_id_adolescents_filtered.csv`; no adults.
- **ROI input:** eight ROIs in `ROIs.csv`.
- **Features:** 825 voxelwise beta features, constructed by pivoting `roi`, `coef`, and within-subject/ROI/coef occurrence index.
- **Outer splits:** 30 specified random seeds, stratified 80% training / 20% test.
- **Feature selection:** ANOVA `SelectKBest(f_classif, k=10)`, fitted **only on training data**.
- **Interactions:** `PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)`.
- **Standardization:** `StandardScaler`, fitted **only on training data**.
- **Random forest:** 600 trees; `max_features='sqrt'`; `max_depth=10`; `min_samples_leaf=12`; `min_samples_split=40`; bootstrap; `class_weight='balanced_subsample'`.
- **Inner CV:** 3 stratified folds in the training split. Only **one fixed model configuration** is evaluated; **this is not hyperparameter tuning**.
- **Metrics:** test accuracy, balanced accuracy, precision, recall, F1, and confusion matrix. Binary precision/recall/F1 use class 1 as positive.
- **SHAP:** omitted because explanations do not change model predictions.

## Run

```bash
python -m pip install -r requirements.txt
python run_experiment.py --seeds 30
```

Quick check:

```bash
python run_experiment.py --seeds 1
```

The script creates `results/efa_per_seed.csv`, `results/efa_confusion.csv`, and `results/efa_labels.csv`. These outputs can be compared against the supplied `efa_previous_30_seeds.csv`.

## Files

- `run_experiment.py` — complete binary-classification reproduction script.
- `data/betas_multi_voxels.csv` — fMRI beta data with `subject_id`, `roi`, `coef`, `beta`.
- `data/ROIs.csv` — selected ROIs.
- `data/adolescents_labels_filtered.csv` — **original binary labels**, `subject_id` and `impulsive`.
- `data/subject_id_adolescents_filtered.csv` — adolescent cohort IDs.
- `results/efa_previous_30_seeds.csv` — previously obtained 30-split test metrics.
- `requirements.txt` — Python dependencies.
- `SHA256SUMS.txt` — file integrity hashes.

## Reproducibility and sharing cautions

The original supplied binary labels are **not regenerated** from factor analysis here. A previously attempted reconstruction produced substantially different labels; therefore, do not substitute regenerated labels without verifying their exact provenance. The observed 68.67% is tied to the supplied original labels and this fixed pipeline.

All supervised preprocessing is fitted inside the training split or inner training fold. However, this is repeated holdout evaluation rather than independent external validation, and it does not establish causal or clinical validity.

**Before publishing publicly:** verify that the study consent, data-use agreements, and institutional rules permit redistribution of participant-level fMRI beta values and identifiers. If not, publish the code and aggregate metrics only and provide approved controlled-access data instructions.
