"""
Models Page — EPA Walkability Index
Author: Sejal Hukare
"""

import streamlit as st
import pca_tab
import clustering_tab
import arm_tab
import nb_tab
import dt_tab
import regression_tab

def app():
    pca_t, clust_t, arm_t, nb_t, dt_t, reg_t = st.tabs([
        "PCA",
        "Clustering",
        "Association Rule Mining",
        "Naive Bayes",
        "Decision Tree",
        "Regression",
    ])

    with pca_t:
        pca_tab.app()
    with clust_t:
        clustering_tab.app()
    with arm_t:
        arm_tab.app()
    with nb_t:
        nb_tab.app()
    with dt_t:
        dt_tab.app()
    with reg_t:
        regression_tab.app()