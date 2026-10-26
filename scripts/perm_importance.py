import os
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

repo_dir = "/home/brooklyn/splicing-ml-project"
os.chdir(repo_dir)

# 1. Load Model & Split Data
model_path = os.path.join(repo_dir, "results/models/r1_winner_xgboost.joblib")
split_path = os.path.join(repo_dir, "results/models/r1_train_test_data_grouped.joblib")
feat_path = os.path.join(repo_dir, "results/features/feature_matrix_v2.csv")

model = joblib.load(model_path)
split_data = joblib.load(split_path)
df_kd1 = pd.read_csv(feat_path)

# 2. 23-feature preparation (matching r1_step5_model_protocol.py)
strand_dummies_kd1 = pd.get_dummies(df_kd1["strand"], prefix="strand", dtype=float)
if "strand_+" not in strand_dummies_kd1.columns: strand_dummies_kd1["strand_+"] = 0.0
if "strand_-" not in strand_dummies_kd1.columns: strand_dummies_kd1["strand_-"] = 0.0
df_kd1 = pd.concat([df_kd1, strand_dummies_kd1[["strand_+", "strand_-"]]], axis=1)

psi_cols = ["psi_control_mean", "psi_control_var", "psi_kd_mean", "psi_kd_var", "dpsi"]
df_kd1[psi_cols] = df_kd1[psi_cols].fillna(0.0)

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

X = df_kd1[feat_23]
y = df_kd1["label"]
test_idx = split_data["test_idx"]

X_test = X.iloc[test_idx].copy().reset_index(drop=True)
y_test = y.iloc[test_idx].copy().reset_index(drop=True)

classes = sorted(list(np.unique(y)))
label_to_idx = {c: i for i, c in enumerate(classes)}
y_test_idx = y_test.map(label_to_idx).values

# 3. Unshuffled baseline
base_pred = model.predict(X_test)
base_f1 = f1_score(y_test_idx, base_pred, average="macro", zero_division=0)
print(f"Unshuffled baseline test Macro-F1: {base_f1:.6f} (reported: 0.6060)\n")

# 4. Define groups
cov_6 = ["cov_mean_control", "cov_var_control", "cov_mean_kd", "cov_var_kd", "cov_mean_overall", "cov_var_overall"]
cov_5 = ["cov_mean_control", "cov_var_control", "cov_mean_kd", "cov_var_kd", "cov_mean_overall"]
psi_5 = ["psi_control_mean", "psi_control_var", "psi_kd_mean", "psi_kd_var", "dpsi"]
utract_4 = ["max_utract_length_upstream", "max_utract_length_downstream", "u_fraction_upstream", "u_fraction_downstream"]
pwm_2 = ["score_5ss_donor", "score_3ss_acceptor"]
gc_3 = ["gc_exon", "gc_upstream_100bp", "gc_downstream_100bp"]
strand_2 = ["strand_+", "strand_-"]

evaluation_items = []
# Single features
for feat in feat_23:
    evaluation_items.append((f"Feature: {feat}", [feat]))

# Groups requested
evaluation_items.append(("Group: exon_length alone", ["exon_length"]))
evaluation_items.append(("Group: 5 coverage features", cov_5))
evaluation_items.append(("Group: 6 coverage features (all)", cov_6))
evaluation_items.append(("Group: 5 PSI/dPSI features", psi_5))
evaluation_items.append(("Group: 4 U-tract features", utract_4))
evaluation_items.append(("Group: 2 PWM features", pwm_2))
evaluation_items.append(("Group: 3 GC features", gc_3))
evaluation_items.append(("Group: 2 strand dummies", strand_2))

results = []
n_rows = len(X_test)

# Seed 42 for permutations
rng = np.random.RandomState(42)

for name, cols in evaluation_items:
    f1_drops = []
    for rep in range(30):
        X_perm = X_test.copy()
        perm_idx = rng.permutation(n_rows)
        # Jointly shuffle the specified columns across test events
        X_perm[cols] = X_perm[cols].iloc[perm_idx].values
        pred = model.predict(X_perm)
        f1_perm = f1_score(y_test_idx, pred, average="macro", zero_division=0)
        drop = base_f1 - f1_perm
        f1_drops.append(drop)
    
    mean_drop = np.mean(f1_drops)
    sd_drop = np.std(f1_drops, ddof=1)
    results.append({
        "Feature / Group": name,
        "N_Features": len(cols),
        "Mean Drop in Macro-F1": mean_drop,
        "SD Drop (ddof=1)": sd_drop,
        "Mean Permuted Macro-F1": base_f1 - mean_drop
    })

df_res = pd.DataFrame(results)
df_res = df_res.sort_values(by="Mean Drop in Macro-F1", ascending=False).reset_index(drop=True)

print("=" * 105)
print("PERMUTATION IMPORTANCE: PRIMARY XGBOOST CLASSIFIER (N=1,807 HELD-OUT TEST EVENTS)")
print("30 Permutations per Row, Random Seed = 42, Baseline Macro-F1 = 0.6060")
print("=" * 105)
print(df_res.to_string(index=False, justify="left"))
