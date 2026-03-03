"""
Models Page — EPA Walkability Index
Author: Sejal Hukare
Delegates to pca_tab.py, clustering_tab.py, arm_tab.py
"""

import streamlit as st
import pca_tab
import clustering_tab
import arm_tab


def app():
    pca_t, clust_t, arm_t = st.tabs([
        "📐  PCA",
        "🔵  Clustering",
        "🔗  Association Rule Mining",
    ])

    with pca_t:
        pca_tab.app()

    with clust_t:
        clustering_tab.app()

    with arm_t:
        arm_tab.app()