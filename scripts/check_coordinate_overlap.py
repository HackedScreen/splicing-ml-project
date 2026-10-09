import joblib
import pandas as pd

df = pd.read_csv("results/features/feature_matrix_v2.csv")
split_grp = joblib.load("results/models/train_test_data_v2_gene_grouped.joblib")
df_train = df.iloc[split_grp["train_idx"]].copy()
df_test = df.iloc[split_grp["test_idx"]].copy()

df_train["coord_str"] = df_train["chrom"].astype(str) + ":" + df_train["exon_start"].astype(str) + "-" + df_train["exon_end"].astype(str) + "(" + df_train["strand"].astype(str) + ")"
df_test["coord_str"] = df_test["chrom"].astype(str) + ":" + df_test["exon_start"].astype(str) + "-" + df_test["exon_end"].astype(str) + "(" + df_test["strand"].astype(str) + ")"

overlap = set(df_train["coord_str"]).intersection(set(df_test["coord_str"]))
print(f"Overlapping coordinate loci count: {len(overlap)}")
for c in overlap:
    tr = df_train[df_train["coord_str"] == c]
    te = df_test[df_test["coord_str"] == c]
    print("=" * 70)
    print(f"Locus: {c}")
    print("Train events:")
    for _, r in tr.iterrows():
        print(f"  event_id={r['event_id']}, label={r['label']}, gene_id={r['gene_id']}, symbol={r.get('gene_symbol', '')}")
    print("Test events:")
    for _, r in te.iterrows():
        print(f"  event_id={r['event_id']}, label={r['label']}, gene_id={r['gene_id']}, symbol={r.get('gene_symbol', '')}")
