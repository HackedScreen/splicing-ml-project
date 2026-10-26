#!/usr/bin/env python3
"""
scripts/generate_dissertation_figures.py
Generates publication-quality charts (300 DPI) for:
1. Random split vs Gene-grouped split comparison (Accuracy & Macro-F1 with CIs)
2. KD2 Seen vs Unseen genes performance with majority baselines
3. Feature ablation 5-fold GroupKFold Macro-F1 chart with error bars
4. End-to-end analytical pipeline data flow diagram
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

def generate_random_vs_grouped_chart():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5), dpi=300)
    
    categories = ['Random Split\n(80/20 Stratified)', 'Gene-Grouped Split\n(80/20 Grouped)']
    x = np.arange(len(categories))
    width = 0.32
    
    # Accuracy values (%)
    xgb_acc = [76.97, 67.68]
    rf_acc = [77.03, 66.24]
    # Gene-grouped XGB 95% CI: [65.25, 70.09] -> error = [2.43, 2.41]
    xgb_acc_err = [[0, 67.68 - 65.25], [0, 70.09 - 67.68]]
    
    rects1 = ax1.bar(x - width/2, xgb_acc, width, label='XGBoost', color='#1f77b4', edgecolor='black', yerr=xgb_acc_err, capsize=4)
    rects2 = ax1.bar(x + width/2, rf_acc, width, label='Random Forest', color='#ff7f0e', edgecolor='black')
    
    ax1.set_ylabel('Test Accuracy (%)', fontsize=11, fontweight='bold')
    ax1.set_title('Classification Accuracy by Split Scheme', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories, fontsize=10)
    ax1.set_ylim(0, 95)
    ax1.axhline(50.58, color='gray', linestyle='--', linewidth=1, label='Majority Baseline (50.58%)')
    ax1.grid(axis='y', linestyle=':', alpha=0.6)
    ax1.legend(loc='lower left', fontsize=9)
    
    for bar in rects1:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 1.2, f'{yval:.2f}%', ha='center', va='bottom', fontsize=8.5, fontweight='bold')
    for bar in rects2:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 1.2, f'{yval:.2f}%', ha='center', va='bottom', fontsize=8.5, fontweight='bold')
        
    # Macro-F1 values
    xgb_f1 = [0.6633, 0.6060]
    rf_f1 = [0.6564, 0.5651]
    # Gene-grouped XGB 95% CI: [0.5703, 0.6382] -> error = [0.0357, 0.0322]
    xgb_f1_err = [[0, 0.6060 - 0.5703], [0, 0.6382 - 0.6060]]
    
    rects3 = ax2.bar(x - width/2, xgb_f1, width, label='XGBoost', color='#1f77b4', edgecolor='black', yerr=xgb_f1_err, capsize=4)
    rects4 = ax2.bar(x + width/2, rf_f1, width, label='Random Forest', color='#ff7f0e', edgecolor='black')
    
    ax2.set_ylabel('Macro-Averaged F1-Score', fontsize=11, fontweight='bold')
    ax2.set_title('Macro-F1 Performance by Split Scheme', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(categories, fontsize=10)
    ax2.set_ylim(0, 0.85)
    ax2.axhline(0.1344, color='gray', linestyle='--', linewidth=1, label='Majority Baseline (0.1344)')
    ax2.grid(axis='y', linestyle=':', alpha=0.6)
    ax2.legend(loc='lower left', fontsize=9)
    
    for bar in rects3:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2.0, yval + 0.015, f'{yval:.4f}', ha='center', va='bottom', fontsize=8.5, fontweight='bold')
    for bar in rects4:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2.0, yval + 0.015, f'{yval:.4f}', ha='center', va='bottom', fontsize=8.5, fontweight='bold')
        
    plt.tight_layout()
    out_path = 'results/models/random_vs_grouped_split_chart.png'
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f'Generated: {out_path}')

def generate_kd2_seen_unseen_chart():
    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=300)
    
    strata = ['Full KD2 Cohort\n(N=7,279)', 'Seen Genes in KD1 Train\n(N=4,711, 64.7%)', 'Unseen Genes\n(N=2,568, 35.3%)']
    x = np.arange(len(strata))
    width = 0.28
    
    model_acc = [72.73, 77.16, 64.60]
    base_acc = [61.81, 62.13, 61.21]
    model_f1 = [66.60, 72.33, 56.63] # in % scale for visual
    
    r1 = ax.bar(x - width, model_acc, width, label='XGBoost Test Accuracy (%)', color='#2ca02c', edgecolor='black')
    r2 = ax.bar(x, base_acc, width, label='Majority Class Baseline Accuracy (%)', color='#aec7e8', edgecolor='black')
    r3 = ax.bar(x + width, model_f1, width, label='XGBoost Macro-F1 (×100)', color='#9467bd', edgecolor='black')
    
    ax.set_ylabel('Metric Value (%)', fontsize=11, fontweight='bold')
    ax.set_title('Evaluation on Second Knockdown Construct (KD2) Stratified by Gene Overlap', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(strata, fontsize=10)
    ax.set_ylim(0, 95)
    ax.grid(axis='y', linestyle=':', alpha=0.6)
    ax.legend(loc='lower right', fontsize=9.5)
    
    for rects in [r1, r2, r3]:
        for bar in rects:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2.0, yval + 1.0, f'{yval:.1f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')
            
    plt.tight_layout()
    out_path = 'results/models/kd2_seen_vs_unseen_chart.png'
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f'Generated: {out_path}')

def generate_ablation_chart():
    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=300)
    
    subsets = [
        '(A) All 23 Features',
        '(B) Minus exon_length (22 feats)',
        '(C) Seq/Struct Only (12 feats)',
        '(D) Seq/Struct Minus exon_length (11 feats)',
        '(E) Coverage/PSI Only (11 feats)'
    ]
    macro_f1 = [0.5879, 0.5073, 0.4367, 0.2964, 0.4105]
    std_f1 = [0.0124, 0.0133, 0.0170, 0.0164, 0.0043]
    
    y_pos = np.arange(len(subsets))
    bars = ax.barh(y_pos, macro_f1, xerr=std_f1, capsize=4, color='#3182bd', edgecolor='black', height=0.55)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(subsets, fontsize=10, fontweight='bold')
    ax.invert_yaxis()
    ax.set_xlabel('5-Fold GroupKFold Macro-Averaged F1-Score (Mean ± SD, ddof=1)', fontsize=11, fontweight='bold')
    ax.set_title('Feature Subset Ablation Analysis (XGBoost Architecture, All 9,247 KD1 Events)', fontsize=12, fontweight='bold')
    ax.set_xlim(0, 0.75)
    ax.grid(axis='x', linestyle=':', alpha=0.6)
    
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.025, bar.get_y() + bar.get_height()/2.0, f'{width:.4f} ± {std_f1[int(bar.get_y()+0.5)]:.4f}', va='center', ha='left', fontsize=9, fontweight='bold')
        
    plt.tight_layout()
    out_path = 'results/models/feature_ablation_chart.png'
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f'Generated: {out_path}')

def generate_data_flow_diagram():
    fig, ax = plt.subplots(figsize=(10, 6.5), dpi=300)
    ax.axis('off')
    
    boxes = [
        ("1. Data Acquisition (ENA: E-MTAB-1147)", "Control (ERR127306–9), KD1 (ERR127302–3), KD2 (ERR127304–5)\nTotal: 238.9M raw paired-end 72 bp reads\n[Kaggle Environment]", 0.5, 0.90, '#d9edf7'),
        ("2. Trimming & Alignment", "Trimmomatic v0.41 (210.4M surviving pairs, 87.1%)\nHISAT2 v2.2.1 linear index (Ensembl 116 GRCh38, 94.3% mapped rate)\n[Kaggle Environment]", 0.5, 0.74, '#d9edf7'),
        ("3. Splicing Calling & Verification", "rMATS v4.3.0 (--variable-read-length, depth-matched 1:1 downsampled Control)\nKD1: 9,247 significant events | KD2: 7,279 events | Lane Control: 3,640 events\n[Local WSL2 Environment]", 0.5, 0.58, '#dff0d8'),
        ("4. Biologically Grounded Feature Extraction", "23 Features: Exon length, GC exon/flanks, PWM 5'/3' splice scores,\n4 HNRNPC U-tract motif features, mean/var read coverage, replicate PSI\n[Local WSL2 Environment]", 0.5, 0.42, '#dff0d8'),
        ("5. Gene-Grouped Train/Test Split (80/20)", "Train: 7,440 events (2,989 genes) | Held-Out Test: 1,807 events (748 genes)\nStrict Zero Gene Leakage between Train and Test sets\n[Local WSL2 Environment]", 0.5, 0.26, '#fcf8e3'),
        ("6. Model Evaluation & Deployment", "Winner XGBoost (CV F1=0.5832) vs RF (CV F1=0.5788)\nHeld-Out Test: 67.68% Acc, 0.6060 Macro-F1 | KD2: 72.73% Acc, 0.6660 Macro-F1\nInteractive Web Application (Streamlit, uncalibrated scores)\n[Local WSL2 Environment]", 0.5, 0.10, '#f2dede')
    ]
    
    for title, text, x, y, bg in boxes:
        bbox_props = dict(boxstyle="round,pad=0.5", fc=bg, ec="black", lw=1.5)
        ax.text(x, y, f"{title}\n{text}", ha="center", va="center", fontsize=9.5, bbox=bbox_props, multialignment='center')
        
    # Draw arrows
    for y_top, y_bot in [(0.84, 0.80), (0.68, 0.64), (0.52, 0.48), (0.36, 0.32), (0.20, 0.16)]:
        ax.annotate('', xy=(0.5, y_bot), xytext=(0.5, y_top),
                    arrowprops=dict(facecolor='black', edgecolor='black', width=1.5, headwidth=7, shrink=0.05))
                    
    ax.set_title("End-to-End Seven-Stage Analytical Data Flow Architecture", fontsize=13, fontweight='bold', pad=15)
    plt.tight_layout()
    out_path = 'results/models/pipeline_data_flow_diagram.png'
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f'Generated: {out_path}')

def main():
    os.makedirs('results/models', exist_ok=True)
    generate_random_vs_grouped_chart()
    generate_kd2_seen_unseen_chart()
    generate_ablation_chart()
    generate_data_flow_diagram()
    print("All custom dissertation figures successfully generated.")

if __name__ == "__main__":
    main()
