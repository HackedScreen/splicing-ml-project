#!/usr/bin/env python3
"""
scripts/make_fig4_5_v3.py

Generates Figure 4.5 (Reliability Curves: Uncalibrated vs Calibrated XGBoost)
Specifications:
- 2x3 grid, 9x6 inches at 300 dpi (2700 x 1800 pixels)
- Axis labels >= 10 pt
- Subplot titles >= 11 pt
- Legend: loc='lower right', framealpha=0.9, fontsize=8
- One-vs-rest per class (A3SS, A5SS, MXE, RI, SE) + 6th subplot for Overall (All Classes Pooled)
- Brier score per class in the legend
- Output: results/figures/fig4_5_reliability_v3.png
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.model_selection import GroupKFold
from xgboost import XGBClassifier
from PIL import Image

repo_dir = "/home/brooklyn/splicing-ml-project"
if not os.path.exists(repo_dir):
    repo_dir = "C:/Users/Brooklyn/Downloads/splicing-ml-project"
os.chdir(repo_dir)

# 1. Load data and features
df = pd.read_csv("results/features/feature_matrix_v2.csv")
split_data = joblib.load("results/models/r1_train_test_data_grouped.joblib")
train_idx = split_data["train_idx"]
test_idx = split_data["test_idx"]

strand_dummies = pd.get_dummies(df["strand"], prefix="strand", dtype=float)
if "strand_+" not in strand_dummies.columns: strand_dummies["strand_+"] = 0.0
if "strand_-" not in strand_dummies.columns: strand_dummies["strand_-"] = 0.0
df = pd.concat([df, strand_dummies[["strand_+", "strand_-"]]], axis=1)

psi_cols = ["psi_control_mean", "psi_control_var", "psi_kd_mean", "psi_kd_var", "dpsi"]
df[psi_cols] = df[psi_cols].fillna(0.0)

feat_23 = [
    "exon_length", "gc_exon", "gc_upstream_100bp", "gc_downstream_100bp",
    "max_utract_length_upstream", "max_utract_length_downstream",
    "u_fraction_upstream", "u_fraction_downstream",
    "score_5ss_donor", "score_3ss_acceptor",
    "cov_mean_control", "cov_var_control", "cov_mean_kd", "cov_var_kd",
    "cov_mean_overall", "cov_var_overall",
    "psi_control_mean", "psi_control_var", "psi_kd_mean", "psi_kd_var",
    "dpsi", "strand_+", "strand_-"
]

X_train = df.iloc[train_idx][feat_23].copy()
y_train = df.iloc[train_idx]["label"].copy()
groups_train = df.iloc[train_idx]["gene_id"].copy()

X_test = df.iloc[test_idx][feat_23].copy()
y_test = df.iloc[test_idx]["label"].copy()

classes = sorted(y_train.unique())
label_to_idx = {c: i for i, c in enumerate(classes)}
y_train_idx = y_train.map(label_to_idx)
y_test_idx = y_test.map(label_to_idx)

# 2. Load uncalibrated winner XGBoost
model = joblib.load("results/models/r1_winner_xgboost.joblib")
winner_test_prob = model.predict_proba(X_test)

# 3. Fit CalibratedClassifierCV using 5-fold GroupKFold on training data
cv_splits_train = list(GroupKFold(n_splits=5).split(X_train, y_train_idx, groups=groups_train))
base_cal = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    random_state=42,
    n_jobs=2,
    eval_metric="mlogloss",
    tree_method="hist"
)
cal_clf = CalibratedClassifierCV(estimator=base_cal, method="sigmoid", cv=cv_splits_train)
cal_clf.fit(X_train, y_train_idx)
cal_test_prob = cal_clf.predict_proba(X_test)

y_test_onehot = pd.get_dummies(y_test)[classes].values

# 4. Create 2x3 Grid Plot (9x6 inches at 300 dpi)
fig, axes = plt.subplots(2, 3, figsize=(9, 6), dpi=300)
axes_flat = axes.flatten()

# Class descriptions
CLASS_NAMES = {
    "A3SS": "Alternative 3' Splice Site",
    "A5SS": "Alternative 5' Splice Site",
    "MXE": "Mutually Exclusive Exons",
    "RI": "Retained Intron",
    "SE": "Skipped Exon"
}

# Panels 1 to 5: One-vs-Rest for each class
for idx, c in enumerate(classes):
    ax = axes_flat[idx]
    y_binary = (y_test.values == c).astype(int)

    frac_pos_uncal, mean_pred_uncal = calibration_curve(y_binary, winner_test_prob[:, idx], n_bins=10, strategy="uniform")
    frac_pos_cal, mean_pred_cal = calibration_curve(y_binary, cal_test_prob[:, idx], n_bins=10, strategy="uniform")

    brier_uncal_c = np.mean((winner_test_prob[:, idx] - y_binary) ** 2)
    brier_cal_c = np.mean((cal_test_prob[:, idx] - y_binary) ** 2)

    ax.plot([0, 1], [0, 1], "k--", label="Perfect Calibration", alpha=0.7, linewidth=1.2)
    ax.plot(mean_pred_uncal, frac_pos_uncal, "s-", color="#1f77b4", linewidth=1.5, markersize=5,
            label=f"Uncal (Brier = {brier_uncal_c:.3f})")
    ax.plot(mean_pred_cal, frac_pos_cal, "o-", color="#2ca02c", linewidth=1.5, markersize=5,
            label=f"Cal (Brier = {brier_cal_c:.3f})")

    ax.set_title(f"Class: {c} ({CLASS_NAMES[c]})", fontsize=11, fontweight="bold", pad=6)
    ax.set_xlabel("Mean Predicted Probability", fontsize=10)
    ax.set_ylabel("Fraction of Positives", fontsize=10)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="lower right", framealpha=0.9, fontsize=8)

# Panel 6: Overall Pooled Multiclass Reliability
ax_pool = axes_flat[5]
y_binary_pool = y_test_onehot.flatten()
uncal_pool = winner_test_prob.flatten()
cal_pool = cal_test_prob.flatten()

frac_pos_uncal_pool, mean_pred_uncal_pool = calibration_curve(y_binary_pool, uncal_pool, n_bins=10, strategy="uniform")
frac_pos_cal_pool, mean_pred_cal_pool = calibration_curve(y_binary_pool, cal_pool, n_bins=10, strategy="uniform")

brier_uncal_pool = np.mean(np.sum((winner_test_prob - y_test_onehot) ** 2, axis=1))
brier_cal_pool = np.mean(np.sum((cal_test_prob - y_test_onehot) ** 2, axis=1))

ax_pool.plot([0, 1], [0, 1], "k--", label="Perfect Calibration", alpha=0.7, linewidth=1.2)
ax_pool.plot(mean_pred_uncal_pool, frac_pos_uncal_pool, "s-", color="#1f77b4", linewidth=1.5, markersize=5,
             label=f"Uncal (Multi Brier = {brier_uncal_pool:.3f})")
ax_pool.plot(mean_pred_cal_pool, frac_pos_cal_pool, "o-", color="#2ca02c", linewidth=1.5, markersize=5,
             label=f"Cal (Multi Brier = {brier_cal_pool:.3f})")

ax_pool.set_title("Overall (All Classes Pooled)", fontsize=11, fontweight="bold", pad=6)
ax_pool.set_xlabel("Mean Predicted Probability", fontsize=10)
ax_pool.set_ylabel("Fraction of Positives", fontsize=10)
ax_pool.set_xlim([-0.02, 1.02])
ax_pool.set_ylim([-0.02, 1.02])
ax_pool.grid(True, linestyle=":", alpha=0.6)
ax_pool.legend(loc="lower right", framealpha=0.9, fontsize=8)

plt.suptitle("Figure 4.5: Reliability Curves & Probability Calibration (Uncalibrated vs Calibrated XGBoost)",
             fontsize=12, fontweight="bold", y=0.995)
plt.tight_layout()

os.makedirs("results/figures", exist_ok=True)
out_path = "results/figures/fig4_5_reliability_v3.png"
plt.savefig(out_path, dpi=300)
plt.close()

# Verify dimensions with PIL
img = Image.open(out_path)
print(f"Generated Figure: {out_path}")
print(f"Image Dimensions: {img.size[0]} x {img.size[1]} pixels (width x height)")
print(f"DPI: {img.info.get('dpi')}")
print(f"File Size: {os.path.getsize(out_path):,} bytes")
