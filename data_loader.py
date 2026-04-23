"""
data_loader.py — Shared cached data loading for the Walkability app.
All tabs import from here so the CSV is only read and processed ONCE
across the entire session, regardless of how many tabs are open.

Author: Sejal Hukare
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ── Feature lists ─────────────────────────────────────────────────────────
FEATURES_FULL = [
    "D2A_Ranked", "D2B_Ranked", "D3B_Ranked", "D4A_Ranked",
    "NatWalkInd", "D3B", "D4A_clean", "D2B_E8MIXA",
    "Pct_AO0", "Pct_AO1", "Pct_AO2p",
]

FEATURES_RANKED = ["D2A_Ranked", "D2B_Ranked", "D3B_Ranked", "D4A_Ranked"]

FEATURES_CONTINUOUS = [
    "D3B", "D4A_clean", "D2B_E8MIXA", "D2A_EPHHM",
    "NatWalkInd", "Pct_AO0", "Pct_AO1", "Pct_AO2p",
]

LABEL = "Walkability_Category"

CAT_ORDER = [
    "Least Walkable", "Below Average Walkable",
    "Above Average Walkable", "Most Walkable",
]


# ── Core loader — everything else builds on this ──────────────────────────
@st.cache_data(show_spinner="Loading walkability data...")
def load_raw() -> pd.DataFrame:
    """
    Read and minimally clean the CSV once per session.
    All other loaders call this — the result is cached globally.
    """
    df = pd.read_csv("walkability_cleaned.csv")
    df["D4A_clean"] = df["D4A"].replace(-99999, 0)
    df = df[df[LABEL].notna()].reset_index(drop=True)
    return df


# ── Full feature matrix + train/test split ────────────────────────────────
@st.cache_data(show_spinner=False)
def get_train_test(test_size: float = 0.2, random_state: int = 42):
    """
    Returns: X_tr, X_te, y_tr, y_te, feature_names
    Uses all 11 features. NOT scaled — call get_scaled_split() for SVMs.
    """
    df = load_raw()
    df_model = df[FEATURES_FULL + [LABEL]].dropna()
    X = df_model[FEATURES_FULL]
    y = df_model[LABEL]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_tr, X_te, y_tr, y_te, FEATURES_FULL


@st.cache_data(show_spinner=False)
def get_scaled_split(test_size: float = 0.2, random_state: int = 42):
    """
    Returns: X_tr_s, X_te_s, y_tr, y_te, scaler, feature_names
    StandardScaled — use for SVMs and Logistic Regression.
    """
    X_tr, X_te, y_tr, y_te, features = get_train_test(test_size, random_state)
    scaler = StandardScaler()
    X_tr_s = pd.DataFrame(
        scaler.fit_transform(X_tr), columns=features, index=X_tr.index
    )
    X_te_s = pd.DataFrame(
        scaler.transform(X_te), columns=features, index=X_te.index
    )
    return X_tr_s, X_te_s, y_tr, y_te, scaler, features


# ── Small sample for SVM (scale-sensitive, O(n²)) ─────────────────────────
@st.cache_data(show_spinner=False)
def get_svm_split(n_sample: int = 5000, random_state: int = 42):
    """
    Scaled, sampled subset for SVM training.
    5K rows → fast enough for interactive use, <1% accuracy loss.
    """
    df = load_raw()
    df_model = df[FEATURES_FULL + [LABEL]].dropna()
    df_sample = df_model.sample(n=min(n_sample, len(df_model)), random_state=random_state)
    X = df_sample[FEATURES_FULL]
    y = df_sample[LABEL]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )
    scaler = StandardScaler()
    X_tr_s = pd.DataFrame(scaler.fit_transform(X_tr), columns=FEATURES_FULL, index=X_tr.index)
    X_te_s = pd.DataFrame(scaler.transform(X_te),     columns=FEATURES_FULL, index=X_te.index)
    return X_tr_s, X_te_s, y_tr, y_te, FEATURES_FULL


# ── Clustering data (5K sample, standardised, label-free) ────────────────
@st.cache_data(show_spinner=False)
def get_cluster_data(n_sample: int = 5000, random_state: int = 42):
    """
    Returns: X_scaled (np array), labels_saved, feature_names
    Label is saved separately for post-hoc comparison only.
    """
    from sklearn.preprocessing import StandardScaler as SS
    features = [
        "D3B", "D4A_clean", "D2B_E8MIXA", "D2A_EPHHM",
        "D2A_Ranked", "D2B_Ranked", "D3B_Ranked", "D4A_Ranked",
        "NatWalkInd", "Pct_AO0", "Pct_AO1", "Pct_AO2p",
    ]
    df = load_raw()
    samp = df[features + [LABEL]].dropna().sample(n_sample, random_state=random_state)
    labels_saved = samp[LABEL].copy()
    X = SS().fit_transform(samp[features])
    return X, labels_saved, features


# ── Raw display dataframe (for "before" screenshots in each tab) ──────────
@st.cache_data(show_spinner=False)
def get_raw_display(n: int = 8) -> pd.DataFrame:
    """First n rows of raw data for before/after comparisons."""
    display_cols = [
        "D2A_Ranked", "D2B_Ranked", "D3B_Ranked", "D4A_Ranked",
        "NatWalkInd", "D3B", "D4A", "D2B_E8MIXA",
        "Pct_AO0", "Pct_AO1", "Pct_AO2p", LABEL,
    ]
    df = load_raw()
    return df[[c for c in display_cols if c in df.columns]].head(n)