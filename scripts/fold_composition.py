import os
import joblib
import pandas as pd
from sklearn.model_selection import GroupKFold

repo_dir = "/home/brooklyn/splicing-ml-project"
os.chdir(repo_dir)

split_path = os.path.join(repo_dir, "results/models/r1_train_test_data_grouped.joblib")
feat_path = os.path.join(repo_dir, "results/features/feature_matrix_v2.csv")

split_data = joblib.load(split_path)
train_idx = split_data["train_idx"]

df_kd1 = pd.read_csv(feat_path)
df_train = df_kd1.iloc[train_idx].copy().reset_index(drop=True)

groups_train = df_train["gene_id"]
y_train = df_train["label"]
classes = ["A3SS", "A5SS", "MXE", "RI", "SE"]

gkf = GroupKFold(n_splits=5)

rows = []
total_val_events = 0

for fold, (t_idx, v_idx) in enumerate(gkf.split(df_train, y_train, groups=groups_train), 1):
    val_y = y_train.iloc[v_idx]
    train_genes = set(groups_train.iloc[t_idx])
    val_genes = set(groups_train.iloc[v_idx])
    shared_genes = len(train_genes.intersection(val_genes))
    counts = val_y.value_counts().to_dict()
    
    row = {
        "Fold": f"Fold {fold}",
        "Val_Events": len(v_idx),
        "Val_Genes": len(val_genes),
        "Shared_Genes": shared_genes,
        "A3SS": counts.get("A3SS", 0),
        "A5SS": counts.get("A5SS", 0),
        "MXE": counts.get("MXE", 0),
        "RI": counts.get("RI", 0),
        "SE": counts.get("SE", 0)
    }
    rows.append(row)
    total_val_events += len(v_idx)

# Summary Total Row
all_train_genes = groups_train.nunique()
train_counts = y_train.value_counts().to_dict()
total_row = {
    "Fold": "Total / Set",
    "Val_Events": len(df_train),
    "Val_Genes": all_train_genes,
    "Shared_Genes": 0,
    "A3SS": train_counts.get("A3SS", 0),
    "A5SS": train_counts.get("A5SS", 0),
    "MXE": train_counts.get("MXE", 0),
    "RI": train_counts.get("RI", 0),
    "SE": train_counts.get("SE", 0)
}
rows.append(total_row)

df_fold = pd.DataFrame(rows)

print("=" * 90)
print("5-FOLD GROUPKFOLD FOLD COMPOSITION (7,440 TRAINING EVENTS ACROSS 2,989 GENES)")
print("Splitting: GroupKFold(n_splits=5, shuffle=False) Grouped by gene_id")
print("=" * 90)
print(df_fold.to_string(index=False, justify="left"))
