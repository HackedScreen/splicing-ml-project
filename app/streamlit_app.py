"""
app/streamlit_app.py

Interactive Alternative Splicing Classification App
BSc Dissertation Project: Machine Learning Classification of RNA Splicing Events
Model: XGBoost Classifier (Biological 23-Feature Architecture with HNRNPC U-Tract Motifs)
Dataset: HNRNPC Knockdown vs Control in HeLa Cells (ArrayExpress E-MTAB-1147)
"""

import os
import io
import sys
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# Ensure app directory is importable
app_dir = os.path.dirname(os.path.abspath(__file__))
if app_dir not in sys.path:
    sys.path.insert(0, app_dir)

from predict_core import (
    predict_events,
    validate_input_df,
    load_model,
    ROBUST_NUMERIC_FEATURES,
    ALL_MODEL_FEATURES,
    CANONICAL_CLASSES,
    CLASS_DESCRIPTIONS,
    IDX_TO_LABEL
)

# -----------------------------------------------------------------------------
# 1. Page Configuration & Custom Theme
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Splicing ML Explorer",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

CLASS_COLORS = {
    "SE": "#2b5c8f",
    "RI": "#d95f02",
    "A3SS": "#7570b3",
    "A5SS": "#e7298a",
    "MXE": "#1b9e77"
}

# -----------------------------------------------------------------------------
# 2. Model Loading (Cached, No Fallbacks)
# -----------------------------------------------------------------------------
@st.cache_resource
def get_app_model():
    return load_model()

try:
    model, model_path = get_app_model()
    classes = CANONICAL_CLASSES
except Exception as e:
    st.error(f"❌ **Fatal Model Error**: Failed to load winning model `r1_winner_xgboost.joblib`:\n\n`{e}`\n\nFallback models have been strictly removed per protocol.")
    st.stop()

# -----------------------------------------------------------------------------
# 3. Sidebar (Always Visible Metadata, Metrics & RI Caveat)
# -----------------------------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/dna-helix.png", width=64)
    st.title("Splicing ML Classifier")
    st.markdown("**Version:** R1 Protocol (Variable-Read-Length, 23 Features)")
    st.markdown("**Primary Model:** `Winning XGBoost Classifier` (`n_est=300, max_depth=6, lr=0.05`)")
    st.markdown(f"**Loaded Model Path:**\n`{model_path}`")
    
    st.markdown("---")
    st.subheader("📊 Held-Out Test Metrics ($N=1,807$)")
    st.markdown("- **XGBoost test 67.68% / 0.606** (macro-F1)")
    st.markdown("- **Majority baseline:** 50.58% accuracy / 0.134 macro-F1 (SE majority)")
    st.markdown("- **Split:** Gene-grouped (0 shared genes)")

    st.markdown("---")
    st.subheader("🌐 Second Knockdown Construct (KD2, $N=7,279$)")
    st.markdown("- **KD2 72.73% / 0.666** (majority baseline 61.81% / 0.153)")
    st.markdown("- **Seen genes:** 77.16% vs 62.13% baseline")
    st.markdown("- **Unseen genes:** (KD2: 64.6% vs 61.2% baseline) (64.60% vs 61.21%)")

    st.markdown("---")
    st.warning(
        "⚠️ **Model Calibration & Probability Scores**:\n\n"
        "Model output probabilities are uncalibrated scores (Brier score: 0.450 uncalibrated vs 0.435 calibrated). "
        "Probabilities represent relative ranking scores rather than calibrated posterior probabilities."
    )

    st.markdown("---")
    st.subheader("🔬 Dataset Scope & Mechanism")
    st.info(
        "**Biological Context:**\n\n"
        "HNRNPC knockdown vs. Control, HeLa cells (*ArrayExpress E-MTAB-1147* / *Zarnack et al., 2013*).\n\n"
        "Augmented with **HNRNPC U-tract binding motif density** in 100 bp flanking splice windows, capturing competitive repression against U2AF65."
    )

    st.caption("BSc Dissertation Project Pipeline • §3–§4 Deployment Demonstration")

# -----------------------------------------------------------------------------
# 4. Main Title & Description
# -----------------------------------------------------------------------------
st.title("🧬 RNA Alternative Splicing Event Classifier")
st.caption(f"🔧 **Active Model File:** `{model_path}`")
st.markdown(
    "Predict the molecular type of human alternative splicing events (`SE`, `RI`, `A5SS`, `A3SS`, `MXE`) "
    "from sequence composition, HNRNPC U-tract binding motifs, splice site consensus strengths, local RNA-seq read coverage, and $\\Psi$ (PSI) kinetics."
)

tab1, tab2 = st.tabs(["📁 Batch Predict (CSV)", "🔍 Single Event Explorer"])

# -----------------------------------------------------------------------------
# TAB 1: Batch Prediction
# -----------------------------------------------------------------------------
with tab1:
    st.subheader("Batch Prediction from CSV File")
    st.markdown(
        "Upload a CSV file containing event feature measurements. The model expects 21 numeric features plus the `strand` (`+` or `-`)."
    )

    # Template generation
    sample_data = {
        "exon_length": [118],
        "gc_exon": [54.0],
        "gc_upstream_100bp": [50.0],
        "gc_downstream_100bp": [50.0],
        "max_utract_length_upstream": [6],
        "max_utract_length_downstream": [4],
        "u_fraction_upstream": [0.34],
        "u_fraction_downstream": [0.30],
        "score_5ss_donor": [4.20],
        "score_3ss_acceptor": [6.22],
        "cov_mean_control": [16.5],
        "cov_var_control": [163.7],
        "cov_mean_kd": [15.7],
        "cov_var_kd": [154.3],
        "cov_mean_overall": [16.1],
        "cov_var_overall": [163.2],
        "psi_control_mean": [0.84],
        "psi_control_var": [0.0003],
        "psi_kd_mean": [0.81],
        "psi_kd_var": [0.0003],
        "dpsi": [0.03],
        "strand": ["+"]
    }
    template_df = pd.DataFrame(sample_data)
    csv_template_bytes = template_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="📥 Download Template CSV (Headers + 1 Example Row)",
        data=csv_template_bytes,
        file_name="splicing_features_v2_template.csv",
        mime="text/csv"
    )

    uploaded_file = st.file_uploader("Upload Event Feature CSV", type=["csv"])

    if uploaded_file is not None:
        try:
            input_df = pd.read_csv(uploaded_file)
            st.write(f"Uploaded **{len(input_df)}** rows. Validating format and feature schema...")

            # Strict batch validation
            validation_warnings = validate_input_df(input_df)
            for w in validation_warnings:
                st.warning(f"⚠️ **Data Warning**: {w}")

            st.success("✅ Validation passed! Running predictions via `predict_core` inference engine...")

            results_df = predict_events(input_df, model=model)

            # Check if any predicted as RI
            ri_predicted_count = (results_df["predicted_class"] == "RI").sum()
            if ri_predicted_count > 0:
                st.warning(
                    f"⚠️ **Note**: **{ri_predicted_count}** of the {len(results_df)} events were classified as **Retained Intron (RI)**. "
                    f"RI predictions exhibit an empirical ~77% F1 (vs 76–81% for SE and MXE) due to sequence similarity with continuous coding sequence. "
                    f"Please verify these loci with independent read-density inspection."
                )

            st.dataframe(
                results_df[[
                    "predicted_class", "predicted_class_name", "confidence",
                    "prob_SE", "prob_RI", "prob_A3SS", "prob_A5SS", "prob_MXE"
                ] + [c for c in input_df.columns if c not in ["predicted_class", "predicted_class_name", "confidence"]]],
                use_container_width=True
            )

            # Download results
            res_csv = results_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Full Batch Predictions (CSV)",
                data=res_csv,
                file_name="splicing_predictions_results.csv",
                mime="text/csv"
            )

        except Exception as ex:
            st.error(f"❌ **Validation / Prediction Error**: {ex}")

# -----------------------------------------------------------------------------
# TAB 2: Single Event Explorer
# -----------------------------------------------------------------------------
with tab2:
    st.subheader("Single Alternative Splicing Event Explorer")
    st.markdown(
        "Adjust sequence, structural, HNRNPC U-tract motif, coverage, and replicate $\\Psi$ parameters to evaluate how the winning XGBoost model "
        "classifies the splicing event."
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("#### 📏 Geometry & Splice Sites")
        exon_len = st.number_input("Exon length (bp)", min_value=1, max_value=20000, value=118, step=1)
        score_5ss = st.slider("5' Splice Site (Donor) PWM", min_value=-16.0, max_value=9.0, value=4.20, step=0.1)
        score_3ss = st.slider("3' Splice Site (Acceptor) PWM", min_value=-16.5, max_value=11.5, value=6.22, step=0.1)
        strand_val = st.selectbox("Genomic Strand", options=["+", "-"], index=0)

    with col2:
        st.markdown("#### 🧬 GC Content Composition (%)")
        gc_ex = st.slider("Exon GC percentage (0–100%)", min_value=0.0, max_value=100.0, value=54.0, step=0.5)
        gc_up = st.slider("Upstream 100bp GC percentage (0–100%)", min_value=0.0, max_value=100.0, value=50.0, step=0.5)
        gc_dn = st.slider("Downstream 100bp GC percentage (0–100%)", min_value=0.0, max_value=100.0, value=50.0, step=0.5)

    with col3:
        st.markdown("#### 🔬 HNRNPC U-Tract Motifs")
        utract_up_len = st.number_input("Upstream Max U-Tract (bp)", min_value=0, max_value=50, value=6, step=1)
        utract_dn_len = st.number_input("Downstream Max U-Tract (bp)", min_value=0, max_value=50, value=4, step=1)
        u_frac_up = st.slider("Upstream U-Fraction (0–1)", min_value=0.0, max_value=1.0, value=0.34, step=0.01)
        u_frac_dn = st.slider("Downstream U-Fraction (0–1)", min_value=0.0, max_value=1.0, value=0.30, step=0.01)

    with col4:
        st.markdown("#### 📈 Coverage Statistics")
        cov_ctrl_m = st.number_input("Control Mean (reads)", min_value=0.0, max_value=2000.0, value=16.5, step=1.0)
        cov_ctrl_v = st.number_input("Control Variance", min_value=0.0, max_value=300000.0, value=163.7, step=10.0)
        cov_kd_m = st.number_input("KD Mean (reads)", min_value=0.0, max_value=2000.0, value=15.7, step=1.0)
        cov_kd_v = st.number_input("KD Variance", min_value=0.0, max_value=300000.0, value=154.3, step=10.0)

    st.markdown("---")
    st.markdown("#### ⚖️ Replicate Inclusion Levels (PSI / $\\Psi$) & Overall Signals")
    col_p1, col_p2, col_p3 = st.columns(3)

    with col_p1:
        psi_ctrl_m = st.slider("Control PSI Mean (0–1)", min_value=0.0, max_value=1.0, value=0.84, step=0.01)
        psi_ctrl_v = st.number_input("Control PSI Variance", min_value=0.0, max_value=0.5, value=0.0003, format="%.4f")
    with col_p2:
        psi_kd_m = st.slider("Knockdown PSI Mean (0–1)", min_value=0.0, max_value=1.0, value=0.81, step=0.01)
        psi_kd_v = st.number_input("Knockdown PSI Variance", min_value=0.0, max_value=0.5, value=0.0003, format="%.4f")
    with col_p3:
        delta_psi = psi_ctrl_m - psi_kd_m
        st.number_input("dPSI = Control - KD", value=float(f"{delta_psi:.4f}"), disabled=True)
        cov_tot_m = (cov_ctrl_m + cov_kd_m) / 2.0
        cov_tot_v = (cov_ctrl_v + cov_kd_v) / 2.0
        st.caption(f"Calculated Overall Coverage Mean: **{cov_tot_m:.2f}**, Variance: **{cov_tot_v:.2f}**")

    if st.button("🔮 Predict Alternative Splicing Event Type", type="primary", use_container_width=True):
        single_row = {
            "exon_length": exon_len,
            "gc_exon": gc_ex,
            "gc_upstream_100bp": gc_up,
            "gc_downstream_100bp": gc_dn,
            "max_utract_length_upstream": utract_up_len,
            "max_utract_length_downstream": utract_dn_len,
            "u_fraction_upstream": u_frac_up,
            "u_fraction_downstream": u_frac_dn,
            "score_5ss_donor": score_5ss,
            "score_3ss_acceptor": score_3ss,
            "cov_mean_control": cov_ctrl_m,
            "cov_var_control": cov_ctrl_v,
            "cov_mean_kd": cov_kd_m,
            "cov_var_kd": cov_kd_v,
            "cov_mean_overall": cov_tot_m,
            "cov_var_overall": cov_tot_v,
            "psi_control_mean": psi_ctrl_m,
            "psi_control_var": psi_ctrl_v,
            "psi_kd_mean": psi_kd_m,
            "psi_kd_var": psi_kd_v,
            "dpsi": delta_psi,
            "strand": strand_val
        }

        single_df = pd.DataFrame([single_row])
        res_single = predict_events(single_df, model=model)
        top_class = res_single["predicted_class"].iloc[0]
        top_p = res_single["confidence"].iloc[0]

        prob_dict = {c: res_single[f"prob_{c}"].iloc[0] for c in classes}
        sorted_probs = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)

        st.markdown("---")
        st.markdown("### 🎯 Prediction Results")

        res_c1, res_c2 = st.columns([1, 2])

        with res_c1:
            st.metric(
                label="Predicted Splicing Event Type",
                value=f"{top_class} ({CLASS_DESCRIPTIONS.get(top_class, '')})",
                delta=f"Confidence: {top_p*100:.1f}%"
            )
            st.markdown(f"**Description:** {CLASS_DESCRIPTIONS.get(top_class, '')}")

            ri_prob = prob_dict.get("RI", 0.0)
            if top_class == "RI" or (top_p - ri_prob <= 0.15):
                st.warning(
                    f"⚠️ **Retained Intron Caveat Triggered**:\n\n"
                    f"RI probability is **{ri_prob*100:.1f}%** (within 15% of top call). "
                    f"Retained Intron events exhibit an empirical **~77% F1** (vs 76–81% for SE and MXE) "
                    f"due to sequence similarity with continuous exons. Treat this call with extra scrutiny."
                )

        with res_c2:
            st.markdown("#### Class Probability Distribution")
            fig, ax = plt.subplots(figsize=(7, 3.5), dpi=300)
            y_pos = np.arange(len(classes))
            bar_colors = [CLASS_COLORS.get(c, "#2b5c8f") for c in classes]
            bars = ax.barh(y_pos, [prob_dict[c] for c in classes], color=bar_colors, edgecolor="black", height=0.6)

            for bar in bars:
                width = bar.get_width()
                ax.text(
                    width + 0.01, bar.get_y() + bar.get_height() / 2,
                    f"{width*100:.1f}%",
                    va="center", ha="left", fontsize=9, fontweight="bold"
                )

            ax.set_yticks(y_pos)
            ax.set_yticklabels([f"{c} ({CLASS_DESCRIPTIONS.get(c, '')})" for c in classes], fontsize=9)
            ax.set_xlim(0, 1.15)
            ax.set_xlabel("Predicted Probability", fontsize=10, fontweight="bold")
            ax.set_title("XGBoost Multi-Class Output Probabilities", fontsize=11, fontweight="bold", pad=10)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
