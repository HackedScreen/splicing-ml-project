import os
import glob
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report

def main():
    base_dir = "."
    models_dir = os.path.join(base_dir, "results", "models")
    features_dir = os.path.join(base_dir, "results", "features")
    lane_dir = os.path.join(base_dir, "results", "splicing_lane_control")

    # =========================================================================
    # ITEM 1: Lane control (ERR127306 vs ERR127307)
    # =========================================================================
    print("=" * 80)
    print("ITEM 1: LANE NEGATIVE CONTROL (|dPSI| DISTRIBUTION FOR FDR < 0.05)")
    print("=" * 80)
    jc_files = glob.glob(os.path.join(lane_dir, "*.MATS.JC.txt"))
    lane_dfs = []
    for f in jc_files:
        etype = os.path.basename(f).split(".")[0]
        df_jc = pd.read_csv(f, sep="\t")
        df_jc["EventType"] = etype
        lane_dfs.append(df_jc)
    df_lane_all = pd.concat(lane_dfs, ignore_index=True)

    sig_lane = df_lane_all[df_lane_all["FDR"] < 0.05].copy()
    print(f"Total significant calls at FDR < 0.05: {len(sig_lane):,}")

    abs_dpsi = sig_lane["IncLevelDifference"].abs()
    med_val = abs_dpsi.median()
    q1 = abs_dpsi.quantile(0.25)
    q3 = abs_dpsi.quantile(0.75)
    iqr_val = q3 - q1
    mean_val = abs_dpsi.mean()
    sd_val = abs_dpsi.std()

    count_ge_01 = (abs_dpsi >= 0.1).sum()
    share_ge_01 = (count_ge_01 / len(abs_dpsi)) * 100.0

    count_ge_05 = (abs_dpsi >= 0.5).sum()
    share_ge_05 = (count_ge_05 / len(abs_dpsi)) * 100.0

    print(f"|dPSI| Median:     {med_val:.4f}")
    print(f"|dPSI| Q1 (25%):   {q1:.4f}")
    print(f"|dPSI| Q3 (75%):   {q3:.4f}")
    print(f"|dPSI| IQR:        {iqr_val:.4f}")
    print(f"|dPSI| Mean ± SD:  {mean_val:.4f} ± {sd_val:.4f}")
    print(f"Share with |dPSI| >= 0.1: {count_ge_01:,} / {len(abs_dpsi):,} ({share_ge_01:.2f}%)")
    print(f"Share with |dPSI| >= 0.5: {count_ge_05:,} / {len(abs_dpsi):,} ({share_ge_05:.2f}%)")

    # Histogram
    plt.figure(figsize=(8, 5))
    plt.hist(abs_dpsi, bins=30, range=(0, 1), color="#4c72b0", edgecolor="black", alpha=0.8)
    plt.axvline(med_val, color="#c44e52", linestyle="--", linewidth=2, label=f"Median = {med_val:.2f}")
    plt.axvline(mean_val, color="#55a868", linestyle=":", linewidth=2, label=f"Mean = {mean_val:.2f}")
    plt.title(f"Lane Negative Control (ERR127306 vs ERR127307)\nDistribution of |ΔPSI| for Significant Events (FDR < 0.05, N = {len(abs_dpsi):,})", fontsize=11, fontweight="bold")
    plt.xlabel("|ΔPSI|", fontsize=11)
    plt.ylabel("Number of Splicing Events", fontsize=11)
    plt.legend(frameon=True, fontsize=10)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    hist_path = os.path.join(models_dir, "lane_control_abs_dpsi_histogram.png")
    plt.savefig(hist_path, dpi=300)
    plt.close()
    print(f"Saved histogram to: {hist_path}")

    # =========================================================================
    # ITEM 2: Class counts per fold for 5-fold GroupKFold & Train/Test
    # =========================================================================
    print("\n" + "=" * 80)
    print("ITEM 2: CLASS COUNTS PER FOLD (5-FOLD GROUPKFOLD) & TRAIN/TEST PARTITIONS")
    print("=" * 80)
    kd1_path = os.path.join(features_dir, "feature_matrix_v2.csv")
    df = pd.read_csv(kd1_path)

    psi_cols = ["psi_control_mean", "psi_control_var", "psi_kd_mean", "psi_kd_var"]
    df[psi_cols] = df[psi_cols].fillna(0.0)

    numeric_features = [
        "exon_length", "gc_exon", "gc_upstream_100bp", "gc_downstream_100bp",
        "max_utract_length_upstream", "max_utract_length_downstream",
        "u_fraction_upstream", "u_fraction_downstream",
        "score_5ss_donor", "score_3ss_acceptor",
        "cov_mean_control", "cov_var_control", "cov_mean_kd", "cov_var_kd",
        "cov_mean_overall", "cov_var_overall",
        "psi_control_mean", "psi_control_var", "psi_kd_mean", "psi_kd_var",
        "dpsi"
    ]
    strand_dummies = pd.get_dummies(df["strand"], prefix="strand", dtype=float)
    if "strand_+" not in strand_dummies.columns: strand_dummies["strand_+"] = 0.0
    if "strand_-" not in strand_dummies.columns: strand_dummies["strand_-"] = 0.0
    strand_dummies = strand_dummies[["strand_+", "strand_-"]]

    X = pd.concat([df[numeric_features], strand_dummies], axis=1)
    y = df["label"]
    groups = df["gene_id"]
    classes = sorted(list(np.unique(y)))

    # Load grouped split
    split_grp = joblib.load(os.path.join(models_dir, "train_test_data_v2_gene_grouped.joblib"))
    train_idx = split_grp["train_idx"]
    test_idx = split_grp["test_idx"]
    y_train_grp = y.iloc[train_idx]
    y_test_grp = y.iloc[test_idx]

    print("\n--- Grouped Train / Test Partitions Class Counts ---")
    df_tt_counts = pd.DataFrame({
        "Full_Dataset": y.value_counts(),
        "Train_N": y_train_grp.value_counts(),
        "Train_Pct": (y_train_grp.value_counts(normalize=True)*100).round(2),
        "Test_N": y_test_grp.value_counts(),
        "Test_Pct": (y_test_grp.value_counts(normalize=True)*100).round(2)
    }).loc[classes]
    print(df_tt_counts.to_string())
    print(f"Total Train: {len(y_train_grp):,} | Total Test: {len(y_test_grp):,}")

    # 5-fold GroupKFold validation class counts
    gkf = GroupKFold(n_splits=5)
    fold_counts = []
    for fold, (t_idx, v_idx) in enumerate(gkf.split(X, y, groups=groups), 1):
        v_y = y.iloc[v_idx]
        counts = v_y.value_counts().to_dict()
        row = {"Fold": f"Fold {fold}", "Total_Events": len(v_idx), "Unique_Genes": groups.iloc[v_idx].nunique()}
        for c in classes:
            row[c] = counts.get(c, 0)
        fold_counts.append(row)
    df_fold_counts = pd.DataFrame(fold_counts)
    print("\n--- 5-Fold GroupKFold Validation Class Counts (Per Fold) ---")
    print(df_fold_counts.to_string(index=False))

    # =========================================================================
    # ITEM 3: Extra Ablations under 5-Fold GroupKFold
    # =========================================================================
    print("\n" + "=" * 80)
    print("ITEM 3: EXTRA ABLATIONS UNDER 5-FOLD GROUPKFOLD")
    print("=" * 80)

    # (i) All features minus exon_length (22 features)
    features_i = [c for c in X.columns if c != "exon_length"]
    # (ii) Sequence-only minus exon_length (11 features)
    seq_struct_features = [
        "gc_exon", "gc_upstream_100bp", "gc_downstream_100bp",
        "score_5ss_donor", "score_3ss_acceptor",
        "max_utract_length_upstream", "max_utract_length_downstream",
        "u_fraction_upstream", "u_fraction_downstream",
        "strand_+", "strand_-"
    ]

    extra_ablations = {
        "(i) All features minus exon_length": features_i,
        "(ii) Sequence-only minus exon_length": seq_struct_features
    }

    for name, feat_list in extra_ablations.items():
        print(f"\nEvaluating: {name} ({len(feat_list)} features)...")
        X_sub = X[feat_list]
        accs, f1s = [], []
        for fold, (t_idx, v_idx) in enumerate(gkf.split(X_sub, y, groups=groups), 1):
            clf = RandomForestClassifier(
                n_estimators=300, max_depth=None, class_weight="balanced", random_state=42, n_jobs=-1
            )
            clf.fit(X_sub.iloc[t_idx], y.iloc[t_idx])
            pred = clf.predict(X_sub.iloc[v_idx])
            accs.append(accuracy_score(y.iloc[v_idx], pred))
            f1s.append(f1_score(y.iloc[v_idx], pred, average="macro", zero_division=0))
        mean_acc, std_acc = np.mean(accs), np.std(accs)
        mean_f1, std_f1 = np.mean(f1s), np.std(f1s)
        print(f"  -> Mean Accuracy: {mean_acc*100:.2f}% ± {std_acc*100:.2f}%")
        print(f"  -> Mean Macro-F1: {mean_f1:.4f} ± {std_f1:.4f}")

    # =========================================================================
    # ITEM 4: DummyClassifier Baselines on Grouped Test Set
    # =========================================================================
    print("\n" + "=" * 80)
    print("ITEM 4: DUMMYCLASSIFIER BASELINES ON GROUPED TEST SET (N = 1,799)")
    print("=" * 80)
    X_train_grp = X.iloc[train_idx]
    X_test_grp = X.iloc[test_idx]

    dummy_mf = DummyClassifier(strategy="most_frequent")
    dummy_mf.fit(X_train_grp, y_train_grp)
    pred_mf = dummy_mf.predict(X_test_grp)
    acc_mf = accuracy_score(y_test_grp, pred_mf)
    f1_mf = f1_score(y_test_grp, pred_mf, average="macro", zero_division=0)
    print(f"DummyClassifier (most_frequent): Accuracy = {acc_mf*100:.2f}%, Macro-F1 = {f1_mf:.4f}")

    dummy_strat = DummyClassifier(strategy="stratified", random_state=42)
    dummy_strat.fit(X_train_grp, y_train_grp)
    pred_strat = dummy_strat.predict(X_test_grp)
    acc_strat = accuracy_score(y_test_grp, pred_strat)
    f1_strat = f1_score(y_test_grp, pred_strat, average="macro", zero_division=0)
    print(f"DummyClassifier (stratified):    Accuracy = {acc_strat*100:.2f}%, Macro-F1 = {f1_strat:.4f}")

    # =========================================================================
    # ITEM 7: Coordinate Overlap Deep-Dive
    # =========================================================================
    print("\n" + "=" * 80)
    print("ITEM 7: TRAIN/TEST COORDINATE OVERLAP DETAILED ANALYSIS")
    print("=" * 80)
    df_train = df.iloc[train_idx].copy()
    df_test = df.iloc[test_idx].copy()

    # Define coordinate string
    coord_cols = ["chr", "strand", "exon_start", "exon_end"]
    df_train["coord_str"] = df_train["chr"].astype(str) + ":" + df_train["exon_start"].astype(str) + "-" + df_train["exon_end"].astype(str) + "(" + df_train["strand"].astype(str) + ")"
    df_test["coord_str"] = df_test["chr"].astype(str) + ":" + df_test["exon_start"].astype(str) + "-" + df_test["exon_end"].astype(str) + "(" + df_test["strand"].astype(str) + ")"

    train_coords = set(df_train["coord_str"])
    test_coords = set(df_test["coord_str"])
    overlap_coords = train_coords.intersection(test_coords)

    print(f"Unique coordinate loci in Train: {len(train_coords):,}")
    print(f"Unique coordinate loci in Test:  {len(test_coords):,}")
    print(f"Overlapping coordinate loci:     {len(overlap_coords)}")

    if len(overlap_coords) > 0:
        for c in overlap_coords:
            tr_events = df_train[df_train["coord_str"] == c]
            te_events = df_test[df_test["coord_str"] == c]
            print(f"\nOverlap Locus: {c}")
            print(f"  Train Events ({len(tr_events)}):")
            for idx, r in tr_events.iterrows():
                print(f"    - Event ID: {r['event_id']}, Label: {r['label']}, Gene: {r['gene_id']} ({r.get('gene_symbol', '')})")
            print(f"  Test Events ({len(te_events)}):")
            for idx, r in te_events.iterrows():
                print(f"    - Event ID: {r['event_id']}, Label: {r['label']}, Gene: {r['gene_id']} ({r.get('gene_symbol', '')})")

if __name__ == "__main__":
    main()
