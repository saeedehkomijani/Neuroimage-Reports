#!/usr/bin/env python3
"""Estimate a pooled latent factor and export participant scores and labels."""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from sklearn.decomposition import FactorAnalysis
from sklearn.preprocessing import StandardScaler

INDICATORS = ["DSM", "Hedonism", "Future Oriented", "Motor", "Self Control", "BART", "DD"]


def analyze(input_file: Path, ids_file: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(input_file, encoding="utf-8-sig")
    if data["subject_id"].isna().any() or not data["subject_id"].is_unique:
        raise ValueError("Participant IDs must be nonmissing and unique.")
    measurements = data[INDICATORS].apply(pd.to_numeric, errors="raise")
    measurements.isna().sum().rename("missing_count").to_csv(
        output_dir / "missingness.csv", header=True
    )
    if measurements.isna().all().any():
        raise ValueError("At least one indicator has no observed values.")
    measurements = measurements.fillna(measurements.mean())
    standardized = StandardScaler().fit_transform(measurements)
    factor = FactorAnalysis(n_components=1, random_state=0)
    scores = factor.fit_transform(standardized).ravel()
    loadings = factor.components_[0].copy()
    if loadings[0] < 0:
        scores, loadings = -scores, -loadings
    cutoff = float(np.median(scores))
    labels = (scores >= cutoff).astype(int)
    all_scores = pd.DataFrame({
        "subject_id": data["subject_id"],
        "factor_score": scores,
        "factor_label": labels,
    })
    all_scores.to_csv(output_dir / "factor_scores_all.csv", index=False)
    pd.DataFrame({
        "indicator": INDICATORS,
        "loading": loadings,
        "uniqueness": factor.noise_variance_,
        "communality": 1 - factor.noise_variance_,
    }).to_csv(output_dir / "factor_loadings.csv", index=False)

    ids = pd.read_csv(ids_file)["subject_id"]
    if ids.isna().any() or not ids.is_unique:
        raise ValueError("Subset IDs must be nonmissing and unique.")
    missing_ids = ids[~ids.isin(data["subject_id"])]
    if not missing_ids.empty:
        raise ValueError(f"Subset IDs not found: {missing_ids.tolist()}")
    subset = all_scores.set_index("subject_id").loc[ids].reset_index()
    subset.to_csv(output_dir / "factor_scores_adolescents.csv", index=False)
    imaging_labels = subset[["subject_id"]].copy()
    imaging_labels["impulsive"] = 1 - subset["factor_label"]
    imaging_labels.to_csv(output_dir / "imaging_labels_adolescents.csv", index=False)
    print(f"Participants: {len(all_scores)}; imaging subset: {len(subset)}")
    print(f"Median factor score: {cutoff:.6f}")
    print(f"Adolescent label counts: {subset['factor_label'].value_counts().sort_index().to_dict()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/impulsivity_factors.csv"))
    parser.add_argument("--ids", type=Path, default=Path("data/adolescent_ids.csv"))
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    args = parser.parse_args()
    analyze(args.input, args.ids, args.output)
