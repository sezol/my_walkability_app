"""
Models Page — EPA Walkability Index
Author: Sejal Hukare
"""

import streamlit as st
import importlib


METHODS = {
    "PCA":                     "pca_tab",
    "Clustering":              "clustering_tab",
    "Association Rule Mining": "arm_tab",
    "Naive Bayes":             "nb_tab",
    "Decision Tree":           "dt_tab",
    "Regression":              "regression_tab",
    "SVM":                     "svm_tab",
    "Ensemble (Random Forest)":"ensemble_tab",
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