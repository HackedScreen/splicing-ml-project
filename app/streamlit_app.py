"""
app/streamlit_app.py

Interactive Alternative Splicing Classification App
BSc Dissertation Project: Machine Learning Classification of RNA Splicing Events
Model: Random Forest Classifier (Robust 19-Feature Architecture)
Dataset: HNRNPC Knockdown vs Control in HeLa Cells (ArrayExpress E-MTAB-1147)
"""

import os
import io
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# 1. Page Configuration & Custom Theme
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Splicing ML Explorer",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define expected features and mappings
ROBUST_NUMERIC_FEATURES = [
    "exon_length",
    "gc_exon",
    "gc_upstream_100bp",
    "gc_downstream_100bp",
    "score_5ss_donor",
    "score_3ss_acceptor",
    "cov_mean_control",
    "cov_var_control",
    "cov_mean_kd",
    "cov_var_kd",
    "cov_mean_overall",
    "cov_var_overall",
    "psi_control_mean",
    "psi_control_var",
    "psi_kd_mean",
    "psi_kd_var",
    "dpsi"
]

ALL_MODEL_FEATURES = ROBUST_NUMERIC_FEATURES + ["strand_+", "strand_-"]

CLASS_DESCRIPTIONS = {
    "SE": "Skipped Exon (Cassette Exon)",
    "RI": "Retained Intron",
    "A3SS": "Alternative 3' Splice Site",
    "A5SS": "Alternative 5' Splice Site",
    "MXE": "Mutually Exclusive Exons"
}

CLASS_COLORS = {
    "SE": "#2b5c8f",
    "RI": "#d95f02",
    "A3SS": "#7570b3",
    "A5SS": "#e7298a",
    "MXE": "#1b9e77"
}

# -----------------------------------------------------------------------------
# 2. Model Loading (Cached)
# -----------------------------------------------------------------------------
@st.cache_resource
def load_robust_model():
    candidates = [
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "models", "random_forest_corrected.joblib"),
        os.path.expanduser("~/splicing-ml-project/results/models/random_forest_corrected.joblib"),
        "results/models/random_forest_corrected.joblib",
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "models", "random_forest_robust.joblib"),
        os.path.expanduser("~/splicing-ml-project/results/models/random_forest_robust.joblib"),
        "results/models/random_forest_robust.joblib"
    ]
    for path in candidates:
        if os.path.exists(path):
            return joblib.load(path), path
    raise FileNotFoundError("Could not locate random_forest_corrected.joblib or random_forest_robust.joblib")

try:
    model, model_path = load_robust_model()
    classes = list(model.classes_)
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

# -----------------------------------------------------------------------------
# 3. Sidebar (Always Visible Metadata, Metrics & RI Caveat)
# -----------------------------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/dna-helix.png", width=64)
    st.title("Splicing ML Classifier")
    st.markdown("**Version:** 1.0 (Robust 19-Feature)")
    st.markdown("**Primary Model:** `Random Forest Ensemble`")
    
    st.markdown("---")
    st.subheader("📊 Held-Out Test Metrics")
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.metric(label="Accuracy", value="74.01%")
        st.metric(label="Macro-F1", value="0.6474")
    with col_m2:
        st.metric(label="ROC-AUC", value="0.9026")
        st.metric(label="Test Size", value="1,739")

    st.markdown("---")
    st.subheader("🔬 Dataset Scope")
    st.info(
        "**Biological Context:**\n\n"
        "HNRNPC knockdown (KD1) vs. Control, HeLa cells (*ArrayExpress E-MTAB-1147* / *Zarnack et al., 2013*).\n\n"
        "Quantified across 8,693 significant alternative splicing events ($FDR < 0.05$) under depth parity."
    )

    st.markdown("---")
    st.warning(
        "⚠️ **Caution on Multi-Class Boundaries**:\n\n"
        "Trained on depth-balanced Control vs KD1. "
        "Alternative splice sites (A5SS, 48% F1; A3SS, 50% F1) show higher boundary ambiguity than Skipped Exons (SE, 79% F1) and Mutually Exclusive Exons (MXE, 76% F1)."
    )

    st.caption("BSc Dissertation Project Pipeline • §3–§4 Deployment Demonstration")

# -----------------------------------------------------------------------------
# 4. Main Title & Description
# -----------------------------------------------------------------------------
st.title("🧬 RNA Alternative Splicing Event Classifier")
st.markdown(
    "Predict the molecular type of human alternative splicing events (`SE`, `RI`, `A5SS`, `A3SS`, `MXE`) "
    "from sequence composition, splice site consensus strengths, local RNA-seq read coverage, and $\\Psi$ (PSI) kinetics."
)

tab1, tab2 = st.tabs(["📁 Batch Predict (CSV)", "🔍 Single Event Explorer"])

# -----------------------------------------------------------------------------
# TAB 1: Batch Prediction
# -----------------------------------------------------------------------------
with tab1:
    st.subheader("Batch Prediction from CSV File")
    st.markdown(
        "Upload a CSV file containing event feature measurements. The model expects 17 numeric features plus the `strand` (`+` or `-`)."
    )

    # Template generation
    sample_data = {
        "exon_length": [118],
        "gc_exon": [0.505],
        "gc_upstream_100bp": [0.44],
        "gc_downstream_100bp": [0.44],
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
        "dpsi": [-0.03],
        "strand": ["+"]
    }
    template_df = pd.DataFrame(sample_data)
    csv_template_bytes = template_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="📥 Download Template CSV (Headers + 1 Example Row)",
        data=csv_template_bytes,
        file_name="splicing_features_template.csv",
        mime="text/csv"
    )

    uploaded_file = st.file_uploader("Upload Event Feature CSV", type=["csv"])

    if uploaded_file is not None:
        try:
            input_df = pd.read_csv(uploaded_file)
            st.write(f"Uploaded **{len(input_df)}** rows. Checking required columns...")

            required_cols = ROBUST_NUMERIC_FEATURES + ["strand"]
            missing_cols = [c for c in required_cols if c not in input_df.columns]

            if missing_cols:
                st.error(
                    f"❌ **Validation Error**: The uploaded CSV is missing the following required column(s):\n\n"
                    f"`{missing_cols}`\n\nPlease check the template format above and re-upload."
                )
            else:
                st.success(" All required columns are present! Formatting features for prediction...")

                # Preprocessing
                proc_df = input_df.copy()
                psi_cols = ["psi_control_mean", "psi_control_var", "psi_kd_mean", "psi_kd_var"]
                proc_df[psi_cols] = proc_df[psi_cols].fillna(0.0)

                # Strand one-hot encoding
                proc_df["strand_+"] = (proc_df["strand"].astype(str).str.strip() == "+").astype(float)
                proc_df["strand_-"] = (proc_df["strand"].astype(str).str.strip() == "-").astype(float)

                X_batch = proc_df[ALL_MODEL_FEATURES]

                # Prediction
                preds = model.predict(X_batch)
                probs = model.predict_proba(X_batch)

                results_df = input_df.copy()
                results_df["predicted_class"] = preds
                results_df["predicted_class_name"] = [CLASS_DESCRIPTIONS.get(p, p) for p in preds]
                results_df["confidence"] = np.max(probs, axis=1)

                for idx, c in enumerate(classes):
                    results_df[f"prob_{c}"] = probs[:, idx]

                # Check if any predicted as RI
                ri_predicted_count = (results_df["predicted_class"] == "RI").sum()
                if ri_predicted_count > 0:
                    st.warning(
                        f"⚠️ **Note**: **{ri_predicted_count}** of the {len(results_df)} events were classified as **Retained Intron (RI)**. "
                        f"RI predictions exhibit an empirical 71% F1 (vs 76–79% for SE and MXE) due to sequence similarity with continuous coding sequence. "
                        f"Please verify these loci with independent read-density inspection."
                    )

                st.dataframe(
                    results_df[[
                        "predicted_class", "predicted_class_name", "confidence",
                        "prob_SE", "prob_RI", "prob_A3SS", "prob_A5SS", "prob_MXE"
                    ] + [c for c in input_df.columns if c not in ["predicted_class"]]],
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
            st.error(f"Error processing file: {ex}")

# -----------------------------------------------------------------------------
# TAB 2: Single Event Explorer
# -----------------------------------------------------------------------------
with tab2:
    st.subheader("Single Alternative Splicing Event Explorer")
    st.markdown(
        "Adjust sequence, structural, coverage, and replicate $\\Psi$ parameters to evaluate how the Random Forest model "
        "classifies the splicing event."
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("#### 📏 Exon Geometry & Splice Sites")
        exon_len = st.number_input("Exon length (bp)", min_value=1, max_value=20000, value=118, step=1)
        score_5ss = st.slider("5' Splice Site (Donor) PWM Score", min_value=-16.0, max_value=9.0, value=4.20, step=0.1)
        score_3ss = st.slider("3' Splice Site (Acceptor) PWM Score", min_value=-16.5, max_value=11.5, value=6.22, step=0.1)
        strand_val = st.selectbox("Genomic Strand", options=["+", "-"], index=0)

    with col2:
        st.markdown("#### 🧬 GC Content Composition")
        gc_ex = st.slider("Exon GC content (0–1)", min_value=0.0, max_value=1.0, value=0.505, step=0.01)
        gc_up = st.slider("Upstream 100bp GC content (0–1)", min_value=0.05, max_value=0.95, value=0.44, step=0.01)
        gc_dn = st.slider("Downstream 100bp GC content (0–1)", min_value=0.05, max_value=0.95, value=0.44, step=0.01)

    with col3:
        st.markdown("#### 📈 RNA-seq Coverage Statistics")
        cov_ctrl_m = st.number_input("Control Coverage Mean (reads)", min_value=0.0, max_value=2000.0, value=16.5, step=1.0)
        cov_ctrl_v = st.number_input("Control Coverage Variance", min_value=0.0, max_value=300000.0, value=163.7, step=10.0)
        cov_kd_m = st.number_input("KD Coverage Mean (reads)", min_value=0.0, max_value=2000.0, value=15.7, step=1.0)
        cov_kd_v = st.number_input("KD Coverage Variance", min_value=0.0, max_value=300000.0, value=154.3, step=10.0)

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
        delta_psi = st.slider("Delta PSI (dPSI = KD - Control)", min_value=-1.0, max_value=1.0, value=-0.03, step=0.01)
        cov_tot_m = (cov_ctrl_m + cov_kd_m) / 2.0
        cov_tot_v = (cov_ctrl_v + cov_kd_v) / 2.0
        st.caption(f"Calculated Overall Coverage Mean: **{cov_tot_m:.2f}**, Variance: **{cov_tot_v:.2f}**")

    if st.button("🔮 Predict Alternative Splicing Event Type", type="primary", use_container_width=True):
        # Format input vector
        single_row = {
            "exon_length": exon_len,
            "gc_exon": gc_ex,
            "gc_upstream_100bp": gc_up,
            "gc_downstream_100bp": gc_dn,
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
            "strand_+": 1.0 if strand_val == "+" else 0.0,
            "strand_-": 1.0 if strand_val == "-" else 0.0
        }

        X_single = pd.DataFrame([single_row])[ALL_MODEL_FEATURES]
        pred_single = model.predict(X_single)[0]
        prob_single = model.predict_proba(X_single)[0]

        prob_dict = dict(zip(classes, prob_single))
        sorted_probs = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)
        top_class, top_p = sorted_probs[0]

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

            # RI Caveat Check: if RI is top prediction OR within 15 points (0.15) of top prediction
            ri_prob = prob_dict.get("RI", 0.0)
            if top_class == "RI" or (top_p - ri_prob <= 0.15):
                st.warning(
                    f"⚠️ **Retained Intron Caveat Triggered**:\n\n"
                    f"RI probability is **{ri_prob*100:.1f}%** (within 15% of top call). "
                    f"Retained Intron events exhibit an empirical **71% F1** (vs 76–79% for SE and MXE) "
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
            ax.set_title("Random Forest Multi-Class Probabilities", fontsize=11, fontweight="bold", pad=10)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
