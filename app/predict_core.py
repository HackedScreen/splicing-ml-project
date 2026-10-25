"""
app/predict_core.py

Core prediction engine and batch preprocessing for alternative splicing classification.
Shared between the Streamlit interactive application (app/streamlit_app.py)
and automated audit regression suites (scripts/test_app_prediction.py).
"""

import os
import joblib
import numpy as np
import pandas as pd

ROBUST_NUMERIC_FEATURES = [
    "exon_length",
    "gc_exon",
    "gc_upstream_100bp",
    "gc_downstream_100bp",
    "max_utract_length_upstream",
    "max_utract_length_downstream",
    "u_fraction_upstream",
    "u_fraction_downstream",
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

CANONICAL_CLASSES = ["A3SS", "A5SS", "MXE", "RI", "SE"]
IDX_TO_LABEL = {i: c for i, c in enumerate(CANONICAL_CLASSES)}
LABEL_TO_IDX = {c: i for i, c in enumerate(CANONICAL_CLASSES)}

CLASS_DESCRIPTIONS = {
    "SE": "Skipped Exon (Cassette Exon)",
    "RI": "Retained Intron",
    "A3SS": "Alternative 3' Splice Site",
    "A5SS": "Alternative 5' Splice Site",
    "MXE": "Mutually Exclusive Exons"
}

_CACHED_MODEL = None
_CACHED_MODEL_PATH = None

def get_model_path():
    candidates = [
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "models", "r1_winner_xgboost.joblib"),
        os.path.expanduser("~/splicing-ml-project/results/models/r1_winner_xgboost.joblib"),
        "results/models/r1_winner_xgboost.joblib",
        "/home/brooklyn/splicing-ml-project/results/models/r1_winner_xgboost.joblib",
        "C:/Users/Brooklyn/Downloads/splicing-ml-project/results/models/r1_winner_xgboost.joblib"
    ]
    for path in candidates:
        if os.path.exists(path):
            return os.path.abspath(path)
    raise FileNotFoundError(
        "Could not locate results/models/r1_winner_xgboost.joblib. "
        "Fallback models have been strictly removed per protocol."
    )

def load_model():
    global _CACHED_MODEL, _CACHED_MODEL_PATH
    if _CACHED_MODEL is None:
        path = get_model_path()
        _CACHED_MODEL = joblib.load(path)
        _CACHED_MODEL_PATH = path
    return _CACHED_MODEL, _CACHED_MODEL_PATH

def validate_input_df(df: pd.DataFrame):
    """
    Validates input dataframe:
    - Required columns
    - Numeric dtypes
    - Rejects NaN outside PSI columns
    - Rejects strand values other than '+' or '-'
    - Returns list of warnings (e.g. if all gc_* <= 1.0)
    """
    warnings = []
    required_cols = ROBUST_NUMERIC_FEATURES + ["strand"]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required column(s): {missing_cols}")

    # Check strand values
    clean_strands = df["strand"].astype(str).str.strip()
    invalid_strands = df.loc[~clean_strands.isin(["+", "-"]), "strand"].unique().tolist()
    if invalid_strands:
        raise ValueError(f"Invalid strand value(s) detected: {invalid_strands}. Strand must strictly be '+' or '-'.")

    # Check numeric columns and convert / check NaNs
    psi_cols = ["psi_control_mean", "psi_control_var", "psi_kd_mean", "psi_kd_var", "dpsi"]
    non_psi_features = [c for c in ROBUST_NUMERIC_FEATURES if c not in psi_cols]

    for col in ROBUST_NUMERIC_FEATURES:
        if not pd.api.types.is_numeric_dtype(df[col]):
            converted = pd.to_numeric(df[col], errors="coerce")
            if converted.isna().sum() > df[col].isna().sum():
                raise TypeError(f"Column '{col}' must contain numeric values only.")

    nan_cols = [c for c in non_psi_features if df[c].isna().any()]
    if nan_cols:
        raise ValueError(f"NaN value(s) detected in non-PSI columns: {nan_cols}. Only PSI columns may contain missing values (imputed as 0.0).")

    # Check GC fraction vs percentage warning
    gc_cols = ["gc_exon", "gc_upstream_100bp", "gc_downstream_100bp"]
    gc_vals = df[gc_cols].apply(pd.to_numeric, errors="coerce")
    if (gc_vals <= 1.0).all().all():
        warnings.append("All GC values (gc_exon, gc_upstream_100bp, gc_downstream_100bp) are <= 1.0. Note: the model was trained on GC percentages (0-100, e.g. 54.0), not fractions (0-1.0).")

    return warnings

def predict_events(df: pd.DataFrame, model=None):
    """
    Applies exact batch preprocessing and runs prediction on event dataframe.
    Returns results_df containing:
    - predicted_class
    - predicted_class_name
    - confidence
    - prob_A3SS, prob_A5SS, prob_MXE, prob_RI, prob_SE
    - original columns
    """
    if model is None:
        model, _ = load_model()

    proc_df = df.copy()

    # Preprocessing
    psi_cols = ["psi_control_mean", "psi_control_var", "psi_kd_mean", "psi_kd_var", "dpsi"]
    proc_df[psi_cols] = proc_df[psi_cols].fillna(0.0)

    # Strand one-hot encoding
    clean_strand = proc_df["strand"].astype(str).str.strip()
    proc_df["strand_+"] = (clean_strand == "+").astype(float)
    proc_df["strand_-"] = (clean_strand == "-").astype(float)

    X_batch = proc_df[ALL_MODEL_FEATURES]

    # Model inference
    raw_preds = model.predict(X_batch)
    is_integer_model = (
        hasattr(model, "classes_") and len(model.classes_) > 0 and isinstance(model.classes_[0], (int, np.integer))
    )
    preds = [IDX_TO_LABEL[p] if is_integer_model else p for p in raw_preds]
    probs = model.predict_proba(X_batch)

    # Class ordering
    if hasattr(model, "classes_"):
        classes_in_model = list(model.classes_)
    else:
        classes_in_model = list(range(len(CANONICAL_CLASSES)))

    results_df = df.copy()
    results_df["predicted_class"] = preds
    results_df["predicted_class_name"] = [CLASS_DESCRIPTIONS.get(p, p) for p in preds]
    results_df["confidence"] = np.max(probs, axis=1)

    for i, c in enumerate(CANONICAL_CLASSES):
        if is_integer_model:
            col_idx = classes_in_model.index(i) if i in classes_in_model else i
        else:
            col_idx = classes_in_model.index(c) if c in classes_in_model else i
        results_df[f"prob_{c}"] = probs[:, col_idx]

    return results_df
