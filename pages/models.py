"""
Models Page — EPA Walkability Index
Author: Sejal Hukare
"""

import streamlit as st
import importlib

METHODS = {
    "PCA":                     "models.pca_tab",
    "Clustering":              "models.clustering_tab",
    "Association Rule Mining": "models.arm_tab",
    "Naive Bayes":             "models.nb_tab",
    "Decision Tree":           "models.dt_tab",
    "Regression":              "models.regression_tab",
    "SVM":                     "models.svm_tab",
    "Ensemble (Random Forest)":"models.ensemble_tab",
}


def app():
    st.markdown("""
    <style>
    div[data-testid="stSelectbox"] > label {
        font-size: 0.85rem; color: #7f8c8d;
        text-transform: uppercase; letter-spacing: 0.05em;
    }
    </style>
    """, unsafe_allow_html=True)

    selected = st.selectbox(
        "Select analysis method",
        list(METHODS.keys()),
        index=0,
        key="model_selector",
    )

    st.markdown("---")

    module = importlib.import_module(METHODS[selected])
    module.app()