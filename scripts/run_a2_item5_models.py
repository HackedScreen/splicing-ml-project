import os
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

print("================================================================================", flush=True)
print("A2 ITEM 5: 5-FOLD GROUPKFOLD MODEL COMPARISON (FROZEN HYPERPARAMETERS)", flush=True)
print("================================================================================", flush=True)

# Load V2 feature matrix
v2_path = "results/features/feature_matrix_v2.csv"
df_v2 = pd.read_csv(v2_path)
groups = df_v2["gene_id"]
y = df_v2["label"]

# One-hot encode strand for df_v2
strand_dummies_v2 = pd.get_dummies(df_v2["strand"], prefix="strand", dtype=float)
if "strand_+" not in strand_dummies_v2.columns:
    strand_dummies_v2["strand_+"] = 0.0
if "strand_-" not in strand_dummies_v2.columns:
    strand_dummies_v2["strand_-"] = 0.0
df_v2 = pd.concat([df_v2, strand_dummies_v2[["strand_+", "strand_-"]]], axis=1)

psi_cols = ["psi_control_mean", "psi_control_var", "psi_kd_mean", "psi_kd_var", "dpsi"]
df_v2[psi_cols] = df_v2[psi_cols].fillna(0.0)

# Define feature sets
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

feat_19 = [
    "exon_length", "gc_exon", "gc_upstream_100bp", "gc_downstream_100bp",
    "score_5ss_donor", "score_3ss_acceptor",
    "cov_mean_control", "cov_var_control", "cov_mean_kd", "cov_var_kd",
    "cov_mean_overall", "cov_var_overall",
    "psi_control_mean", "psi_control_var", "psi_kd_mean", "psi_kd_var",
    "dpsi", "strand_+", "strand_-"
]

# Load MaxEnt feature matrix
maxent_path = "results/features/feature_matrix_v3_maxent.csv"
df_maxent = pd.read_csv(maxent_path)
strand_dummies_maxent = pd.get_dummies(df_maxent["strand"], prefix="strand", dtype=float)
if "strand_+" not in strand_dummies_maxent.columns:
    strand_dummies_maxent["strand_+"] = 0.0
if "strand_-" not in strand_dummies_maxent.columns:
    strand_dummies_maxent["strand_-"] = 0.0
df_maxent = pd.concat([df_maxent, strand_dummies_maxent[["strand_+", "strand_-"]]], axis=1)
df_maxent[psi_cols] = df_maxent[psi_cols].fillna(0.0)

# Label encoding for XGBoost
label_map = {lbl: i for i, lbl in enumerate(sorted(y.unique()))}
y_num = y.map(label_map)

# Models dictionary
models_spec = [
    ("Logistic Regression (V2 23 feats)", "LR", 23, df_v2[feat_23], y),
    ("Random Forest Baseline (19 feats)", "RF-19", 19, df_v2[feat_19], y),
    ("Random Forest V2 (23 feats)", "RF-V2", 23, df_v2[feat_23], y),
    ("XGBoost V2 (23 feats)", "XGB-V2", 23, df_v2[feat_23], y_num),
    ("Random Forest MaxEntScan (23 feats)", "RF-MaxEnt", 23, df_maxent[feat_23], y)
]

gkf = GroupKFold(n_splits=5)
results_summary = []

for model_title, model_id, n_feats, X_mat, y_target in models_spec:
    print(f"\n--- {model_title} (n_features = {n_feats}) ---", flush=True)
    fold_accs = []
    fold_f1s = []
    
    for fold, (t_idx, v_idx) in enumerate(gkf.split(X_mat, y_target, groups=groups), 1):
        X_tr, X_val = X_mat.iloc[t_idx], X_mat.iloc[v_idx]
        y_tr, y_val = y_target.iloc[t_idx], y_target.iloc[v_idx]
        
        if model_id == "LR":
            scaler = StandardScaler()
            X_tr_sc = scaler.fit_transform(X_tr)
            X_val_sc = scaler.transform(X_val)
            clf = LogisticRegression(C=10, max_iter=2000, class_weight='balanced', random_state=42)
            clf.fit(X_tr_sc, y_tr)
            preds = clf.predict(X_val_sc)
        elif model_id == "RF-19":
            clf = RandomForestClassifier(n_estimators=300, max_depth=None, class_weight='balanced', random_state=42, n_jobs=2)
            clf.fit(X_tr, y_tr)
            preds = clf.predict(X_val)
        elif model_id == "RF-V2":
            clf = RandomForestClassifier(n_estimators=300, max_depth=None, class_weight='balanced', random_state=42, n_jobs=2)
            clf.fit(X_tr, y_tr)
            preds = clf.predict(X_val)
        elif model_id == "XGB-V2":
            clf = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05, random_state=42, n_jobs=2, eval_metric='mlogloss', tree_method='hist')
            sample_w = compute_sample_weight("balanced", y_tr)
            clf.fit(X_tr, y_tr, sample_weight=sample_w)
            preds = clf.predict(X_val)
        elif model_id == "RF-MaxEnt":
            clf = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42, n_jobs=2)
            clf.fit(X_tr, y_tr)
            preds = clf.predict(X_val)
            
        acc = accuracy_score(y_val, preds)
        f1 = f1_score(y_val, preds, average="macro", zero_division=0)
        fold_accs.append(acc)
        fold_f1s.append(f1)
        row_fold = {
            "model_name": model_title,
            "model_id": model_id,
            "n_features": n_feats,
            "fold": fold,
            "accuracy": acc,
            "macro_f1": f1
        }
        results_summary.append(row_fold)
        print(f"  Fold {fold}: Acc = {acc*100:.2f}%, Macro-F1 = {f1:.4f}", flush=True)
        
    mean_acc = np.mean(fold_accs)
    sd_acc_ddof1 = np.std(fold_accs, ddof=1)
    mean_f1 = np.mean(fold_f1s)
    sd_f1_ddof1 = np.std(fold_f1s, ddof=1)
    
    print(f"  [Summary] Mean Acc: {mean_acc*100:.2f}% ± {sd_acc_ddof1*100:.2f}% (ddof=1)", flush=True)
    print(f"  [Summary] Mean Macro-F1: {mean_f1:.4f} ± {sd_f1_ddof1:.4f} (ddof=1)", flush=True)

df_res = pd.DataFrame(results_summary)
df_res.to_csv("results/models/a2_item5_groupkfold_model_comparison.csv", index=False)
print("\nSaved 5-fold GroupKFold model comparison to results/models/a2_item5_groupkfold_model_comparison.csv", flush=True)

# Generate markdown summary
summary_rows = []
for (m_title, m_id, n_f), grp in df_res.groupby(["model_name", "model_id", "n_features"], sort=False):
    accs = grp["accuracy"].values
    f1s = grp["macro_f1"].values
    mean_acc = np.mean(accs)
    sd_acc = np.std(accs, ddof=1)
    mean_f1 = np.mean(f1s)
    sd_f1 = np.std(f1s, ddof=1)
    summary_rows.append({
        "Model": m_title,
        "ID": m_id,
        "Features": n_f,
        "Fold 1 Acc": f"{accs[0]*100:.2f}%",
        "Fold 2 Acc": f"{accs[1]*100:.2f}%",
        "Fold 3 Acc": f"{accs[2]*100:.2f}%",
        "Fold 4 Acc": f"{accs[3]*100:.2f}%",
        "Fold 5 Acc": f"{accs[4]*100:.2f}%",
        "Mean Acc ± SD (ddof=1)": f"{mean_acc*100:.2f}% ± {sd_acc*100:.2f}%",
        "Fold 1 F1": f"{f1s[0]:.4f}",
        "Fold 2 F1": f"{f1s[1]:.4f}",
        "Fold 3 F1": f"{f1s[2]:.4f}",
        "Fold 4 F1": f"{f1s[3]:.4f}",
        "Fold 5 F1": f"{f1s[4]:.4f}",
        "Mean Macro-F1 ± SD (ddof=1)": f"{mean_f1:.4f} ± {sd_f1:.4f}"
    })

md_lines = [
    "# 5-Fold GroupKFold Model Comparison (Frozen Hyperparameters)",
    "",
    "| Model | Features | Fold 1 Acc | Fold 2 Acc | Fold 3 Acc | Fold 4 Acc | Fold 5 Acc | Mean Acc ± SD (ddof=1) | Fold 1 F1 | Fold 2 F1 | Fold 3 F1 | Fold 4 F1 | Fold 5 F1 | Mean Macro-F1 ± SD (ddof=1) |",
    "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |"
]
for r in summary_rows:
    md_lines.append(f"| {r['Model']} | {r['Features']} | {r['Fold 1 Acc']} | {r['Fold 2 Acc']} | {r['Fold 3 Acc']} | {r['Fold 4 Acc']} | {r['Fold 5 Acc']} | **{r['Mean Acc ± SD (ddof=1)']}** | {r['Fold 1 F1']} | {r['Fold 2 F1']} | {r['Fold 3 F1']} | {r['Fold 4 F1']} | {r['Fold 5 F1']} | **{r['Mean Macro-F1 ± SD (ddof=1)']}** |")

with open("results/models/a2_item5_groupkfold_model_comparison.md", "w") as f:
    f.write("\n".join(md_lines) + "\n")
print("Saved markdown comparison to results/models/a2_item5_groupkfold_model_comparison.md", flush=True)
