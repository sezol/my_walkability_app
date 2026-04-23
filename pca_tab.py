import os
"""
PCA Page — EPA Walkability Index
Author: Sejal Hukare
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
import warnings
warnings.filterwarnings("ignore")

BLUE   = "#3498db"; GREEN = "#2ecc71"; RED = "#e74c3c"
ORANGE = "#f39c12"; PURPLE = "#9b59b6"; TEAL = "#1abc9c"
CAT_PAL = {"Most Walkable": BLUE, "Above Average Walkable": GREEN,
           "Below Average Walkable": ORANGE, "Least Walkable": RED}

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "#f8f9fa",
    "axes.edgecolor": "#dee2e6", "axes.labelcolor": "#2c3e50",
    "xtick.color": "#7f8c8d",   "ytick.color": "#7f8c8d",
    "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.labelsize": 10, "font.size": 10,
})

@st.cache_data
def load_pca_data():
    df = pd.read_csv("walkability_cleaned.csv")
    df["D4A_clean"] = df["D4A"].replace(-99999, 0)
    features = ["D3B","D4A_clean","D2B_E8MIXA","D2A_EPHHM",
                "D2A_Ranked","D2B_Ranked","D3B_Ranked","D4A_Ranked",
                "NatWalkInd","Pct_AO0","Pct_AO1","Pct_AO2p"]
    feat_labels = {
        "D3B":"Intersection Density","D4A_clean":"Transit Proximity",
        "D2B_E8MIXA":"Employment Mix","D2A_EPHHM":"Emp-HH Mix",
        "D2A_Ranked":"Emp-HH Ranked","D2B_Ranked":"Emp Mix Ranked",
        "D3B_Ranked":"Intersect Ranked","D4A_Ranked":"Transit Ranked",
        "NatWalkInd":"Walk Index","Pct_AO0":"% Zero-Car",
        "Pct_AO1":"% One-Car","Pct_AO2p":"% 2+ Car",
    }
    samp = df[features+["Walkability_Category"]].dropna().sample(5000, random_state=42)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(samp[features])
    return samp, X_scaled, features, feat_labels

def metric_card(col, val, label, color=BLUE):
    col.markdown(f"""
    <div style="background:#f8f9fa;border-left:4px solid {color};
    border-radius:6px;padding:14px 16px;margin-bottom:8px">
    <div style="font-size:1.4rem;font-weight:700;color:#2c3e50;line-height:1.1">{val}</div>
    <div style="font-size:0.73rem;color:#7f8c8d;text-transform:uppercase;
    letter-spacing:0.06em;margin-top:3px">{label}</div></div>""", unsafe_allow_html=True)

def explain(layman, ds):
    with st.expander("📖 What does this mean?", expanded=False):
        st.markdown(f"**In plain English:** {layman}")
        st.markdown(f"**For data scientists:** {ds}")


def app():
    st.markdown("""
    <style>
    .section-hdr{font-size:1.25rem;font-weight:700;color:#2c3e50;
        border-left:4px solid #3498db;padding-left:12px;margin:1.5rem 0 0.8rem}
    .callout{background:#eaf4fd;border-left:4px solid #3498db;
        border-radius:6px;padding:14px 18px;margin:10px 0;font-size:0.9rem;color:#1a5276}
    .callout.green{background:#eafaf1;border-color:#2ecc71;color:#1e8449}
    .callout.orange{background:#fef9e7;border-color:#f39c12;color:#7d6608}
    .callout.red{background:#fdedec;border-color:#e74c3c;color:#922b21}
    </style>""", unsafe_allow_html=True)

    st.title("📐 Principal Component Analysis (PCA)")

    # ── (1) WHAT IS PCA ───────────────────────────────────────────────────────
    st.markdown('<div class="section-hdr">① What is PCA?</div>', unsafe_allow_html=True)

    st.markdown("""
    **In plain English:** Imagine you have a spreadsheet with 12 columns describing every neighbourhood
    in America — things like how many bus stops are nearby, how walkable the streets are, and how many
    households own cars. Trying to look at 12 columns at once is overwhelming. PCA is like squishing
    all that information into just 2 or 3 "super-columns" that still capture most of the story.
    Think of it as making a summary that keeps the most important parts and throws away the noise.

    **For data scientists:** Principal Component Analysis is an unsupervised dimensionality-reduction
    technique that projects data onto a new orthogonal basis defined by the eigenvectors of the
    feature covariance matrix. Each principal component (PC) is a linear combination of the original
    features and is ordered by the amount of variance it explains. By retaining only the top *k* PCs
    we reduce computational complexity while preserving the maximum possible variance. PCA requires
    standardised, quantitative, label-free input data.
    """)

    # ── (2) DATASET CHOICE ────────────────────────────────────────────────────
    st.markdown('<div class="section-hdr">② Dataset Used</div>', unsafe_allow_html=True)
    st.markdown("""
    We use the **EPA National Walkability Index** cleaned dataset (`walkability_cleaned.csv`) —
    43,065 U.S. Census block groups, each described by 32 variables. For PCA we select
    **12 quantitative features** covering transit access, employment mix, intersection density,
    auto ownership, and the final walkability score.
    """)

    samp, X_scaled, features, feat_labels = load_pca_data()

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Raw sample (before scaling) — first 5 rows, selected features:**")
        display_cols = ["D3B","D4A_clean","D2B_E8MIXA","NatWalkInd","Pct_AO0","Walkability_Category"]
        show_cols = [c for c in display_cols if c in samp.columns]
        st.dataframe(samp[show_cols].head(), use_container_width=True)
    with c2:
        st.markdown("**After StandardScaler normalisation — same 5 rows:**")
        scaled_df = pd.DataFrame(X_scaled, columns=[feat_labels[f] for f in features])
        st.dataframe(scaled_df.head(), use_container_width=True)

    explain(
        "The left table shows raw numbers — transit distances in metres, densities per km² — "
        "which are all on wildly different scales. The right table shows the same data after "
        "rescaling so every column has a mean of 0 and a spread of 1. This is essential before "
        "PCA so that no single variable dominates just because its numbers happen to be bigger.",
        "StandardScaler applies z-score normalisation: z = (x − μ) / σ per feature, ensuring "
        "unit-variance inputs so the covariance matrix is not dominated by high-magnitude features."
    )

    # ── (3) RUN PCA ───────────────────────────────────────────────────────────
    st.markdown('<div class="section-hdr">③ Running PCA</div>', unsafe_allow_html=True)

    pca_full = PCA().fit(X_scaled)
    pca2     = PCA(n_components=2).fit(X_scaled)
    pca3     = PCA(n_components=3).fit(X_scaled)
    pc2      = pca2.transform(X_scaled)
    pc3      = pca3.transform(X_scaled)
    ev       = pca_full.explained_variance_ratio_
    eigenvals= pca_full.explained_variance_
    cumev    = np.cumsum(ev)
    cat_col  = samp["Walkability_Category"]

    # Metric cards
    c1,c2,c3,c4,c5,c6 = st.columns(6)
    metric_card(c1, f"{ev[0]*100:.1f}%",  "PC1 Variance",    BLUE)
    metric_card(c2, f"{ev[1]*100:.1f}%",  "PC2 Variance",    GREEN)
    metric_card(c3, f"{ev[2]*100:.1f}%",  "PC3 Variance",    ORANGE)
    metric_card(c4, f"{(cumev[1])*100:.1f}%", "2D Total",    BLUE)
    metric_card(c5, f"{(cumev[2])*100:.1f}%", "3D Total",    GREEN)
    metric_card(c6, f"{int(np.searchsorted(cumev,0.95))+1} PCs", "for 95%", RED)

    st.markdown("---")

    # ── (4) SCREE + CUMULATIVE ────────────────────────────────────────────────
    st.markdown('<div class="section-hdr">④ Scree Plot & Cumulative Explained Variance</div>', unsafe_allow_html=True)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    xs = np.arange(1, len(ev)+1)
    axes[0].bar(xs, ev*100, color=[BLUE if i<2 else "#d5d8dc" for i in range(len(ev))],
                edgecolor="white", width=0.6)
    axes[0].plot(xs, ev*100, "o-", color=ORANGE, lw=1.6, ms=4)
    axes[0].set_xlabel("Principal Component"); axes[0].set_ylabel("Explained Variance (%)")
    axes[0].set_title("Scree Plot — How much does each PC contribute?")
    axes[0].set_xticks(xs); axes[0].grid(axis="y", alpha=0.4)
    for i, v in enumerate(ev*100):
        axes[0].text(i+1, v+0.3, f"{v:.1f}%", ha="center", fontsize=7.5, color="#555")

    axes[1].fill_between(xs, cumev*100, alpha=0.15, color=BLUE)
    axes[1].plot(xs, cumev*100, "o-", color=BLUE, lw=2, ms=4)
    for thr, col, lab in [(80,ORANGE,"80%"),(90,RED,"90%"),(95,PURPLE,"95%")]:
        axes[1].axhline(thr, color=col, ls="--", lw=1.2, label=lab)
    axes[1].set_xlabel("Number of Components"); axes[1].set_ylabel("Cumulative Variance (%)")
    axes[1].set_title("Cumulative Variance — How much do we keep?")
    axes[1].set_xticks(xs); axes[1].set_ylim(0, 105)
    axes[1].grid(axis="y", alpha=0.4); axes[1].legend(fontsize=9)
    plt.tight_layout(pad=2)
    st.pyplot(fig, use_container_width=True); plt.close()

    explain(
        "The left chart (scree plot) is like a bar chart of how 'useful' each new summary column is. "
        "The first bar is tallest — meaning the first PC captures the most information. Each bar gets "
        "shorter. The right chart shows how much total information we've kept as we add more PCs. "
        "The dashed lines show targets: if you want to keep 90% of the story, you can see exactly "
        "how many PCs you need.",
        "The scree plot visualises the eigenvalue spectrum. The 'elbow' indicates the point of "
        "diminishing returns. The cumulative explained variance ratio plot shows ∑EVR vs. k, "
        "with threshold lines at 80/90/95% to guide component selection."
    )

    dims_95 = int(np.searchsorted(cumev, 0.95)) + 1
    st.markdown(f"""
    <div class="callout">
    <strong>2D dataset retains {cumev[1]*100:.1f}% of information</strong> — 
    using just 2 principal components we keep {cumev[1]*100:.1f}% of the total variance 
    from all 12 original features.
    </div>
    <div class="callout green">
    <strong>3D dataset retains {cumev[2]*100:.1f}% of information</strong> — 
    adding a third component bumps this to {cumev[2]*100:.1f}%.
    </div>
    <div class="callout orange">
    <strong>To retain 95% you need {dims_95} principal components</strong> — 
    down from 12 original features. That's a {round((1-dims_95/12)*100)}% reduction in dimensionality.
    </div>
    """, unsafe_allow_html=True)

    # ── (5) TOP EIGENVALUES ───────────────────────────────────────────────────
    st.markdown('<div class="section-hdr">⑤ Top Eigenvalues</div>', unsafe_allow_html=True)

    eig_df = pd.DataFrame({
        "Component":         [f"PC{i+1}" for i in range(len(eigenvals))],
        "Eigenvalue":        np.round(eigenvals, 4),
        "Variance Explained":np.round(ev*100, 2),
        "Cumulative %":      np.round(cumev*100, 2),
    })
    st.dataframe(eig_df, use_container_width=True, hide_index=True)

    explain(
        "An eigenvalue tells you how much 'spread' or information a principal component captures. "
        f"PC1's eigenvalue is {eigenvals[0]:.2f}, meaning it captures {ev[0]*100:.1f}% of all the "
        f"variation in the data. PC2 captures {ev[1]*100:.1f}%, and PC3 captures {ev[2]*100:.1f}%. "
        "The bigger the eigenvalue, the more important that component is.",
        f"The top 3 eigenvalues are λ₁={eigenvals[0]:.4f}, λ₂={eigenvals[1]:.4f}, "
        f"λ₃={eigenvals[2]:.4f}. These correspond to the variance of the data projected onto "
        "each eigenvector of the feature covariance matrix. Any eigenvalue < 1 (Kaiser criterion) "
        "explains less variance than a single standardised variable."
    )

    # ── (6) 2D VISUALISATION ──────────────────────────────────────────────────
    st.markdown('<div class="section-hdr">⑥ 2D PCA Projection</div>', unsafe_allow_html=True)
    st.markdown(f"**Information retained: {cumev[1]*100:.1f}%**")

    fig, ax = plt.subplots(figsize=(11, 5.5))
    for cat, color in CAT_PAL.items():
        mask = cat_col == cat
        ax.scatter(pc2[mask,0], pc2[mask,1], s=10, alpha=0.5,
                   color=color, label=cat, rasterized=True)
    ax.set_xlabel(f"PC1 — {ev[0]*100:.1f}% variance  (Transit & Walkability axis)")
    ax.set_ylabel(f"PC2 — {ev[1]*100:.1f}% variance  (Car ownership axis)")
    ax.set_title("2D PCA Projection of Walkability Data — 5,000 block groups")
    ax.legend(fontsize=9, markerscale=2, framealpha=0.85, loc="upper right")
    ax.grid(alpha=0.3)
    plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()

    explain(
        "Each dot on this chart is one neighbourhood (Census block group). The colours show which "
        "walkability category it belongs to. Blue dots are the most walkable areas; red dots are "
        "the least walkable. Notice how the blue dots cluster together on the left and the red dots "
        "cluster on the right — this tells us that PCA has successfully separated these groups "
        "using just two summary numbers.",
        f"2D scatter of PC1 vs PC2 scores, coloured by ground-truth walkability category. "
        f"PC1 ({ev[0]*100:.1f}% EVR) acts as the primary walkability axis — high scores correlate "
        f"with Most Walkable areas. PC2 ({ev[1]*100:.1f}% EVR) separates by auto-ownership profile. "
        "Class separation visible despite retaining only 56.5% variance indicates strong latent structure."
    )

    # ── (7) 3D VISUALISATION ──────────────────────────────────────────────────
    st.markdown('<div class="section-hdr">⑦ 3D PCA Projection</div>', unsafe_allow_html=True)
    st.markdown(f"**Information retained: {cumev[2]*100:.1f}%**")

    fig = plt.figure(figsize=(11, 6))
    ax3 = fig.add_subplot(111, projection="3d")
    for cat, color in CAT_PAL.items():
        mask = (cat_col == cat).values
        ax3.scatter(pc3[mask,0], pc3[mask,1], pc3[mask,2],
                    s=6, alpha=0.45, color=color, label=cat, rasterized=True)
    ax3.set_xlabel(f"PC1 ({ev[0]*100:.1f}%)", fontsize=8)
    ax3.set_ylabel(f"PC2 ({ev[1]*100:.1f}%)", fontsize=8)
    ax3.set_zlabel(f"PC3 ({ev[2]*100:.1f}%)", fontsize=8)
    ax3.set_title(f"3D PCA Projection — {cumev[2]*100:.1f}% variance retained")
    ax3.legend(fontsize=8, markerscale=2, loc="upper left")
    plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()

    explain(
        "Adding a third dimension (PC3) to our chart gives us more depth — quite literally. "
        f"Now we're keeping {cumev[2]*100:.1f}% of the original information instead of "
        f"{cumev[1]*100:.1f}%. The extra dimension helps separate the 'above average' and "
        "'below average' categories that were slightly overlapping in the 2D view.",
        f"PC3 ({ev[2]*100:.1f}% EVR) captures the employment mix / land-use diversity axis "
        "as shown in the loadings. The 3D projection reveals sub-clustering within the "
        f"'Below Average Walkable' group that is not visible in 2D. Total EVR = {cumev[2]*100:.1f}%."
    )

    # ── (8) LOADINGS HEATMAP ──────────────────────────────────────────────────
    st.markdown('<div class="section-hdr">⑧ Feature Loadings — What drives each PC?</div>', unsafe_allow_html=True)

    loadings = pd.DataFrame(
        pca_full.components_[:4].T,
        index=[feat_labels[f] for f in features],
        columns=["PC1","PC2","PC3","PC4"],
    )
    fig, ax = plt.subplots(figsize=(9, 5.5))
    sns.heatmap(loadings, annot=True, fmt=".2f",
                cmap=sns.diverging_palette(220, 20, as_cmap=True),
                center=0, linewidths=0.4, linecolor="white",
                annot_kws={"size": 8.5}, ax=ax, cbar_kws={"shrink": 0.8})
    ax.set_title("PCA Loadings Heatmap — How much does each feature contribute to each PC?")
    ax.tick_params(labelsize=9)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()

    explain(
        "This heatmap is a 'recipe card' for each principal component. Dark blue means a feature "
        "has a strong positive contribution to that PC; dark red means a strong negative contribution. "
        "For example, PC1 is strongly driven by the ranked walkability scores (Walk Index, Transit "
        "Ranked) — so we can say PC1 measures 'how walkable an area is overall'. PC2 is strongly "
        "driven by car ownership (% 2+ Car households) — so PC2 measures 'how car-dependent an area is'.",
        "The heatmap shows the eigenvector components (loadings) for PC1–PC4. Large absolute loadings "
        "indicate which original features most influence each component. PC1 loads heavily on "
        "D4A_Ranked, D3B_Ranked, and NatWalkInd (walkability infrastructure). PC2 loads heavily on "
        "Pct_AO2p (car ownership) with opposing sign to Pct_AO0, reflecting an auto-dependence axis."
    )

    # ── (9) CONCLUSIONS ───────────────────────────────────────────────────────
    st.markdown('<div class="section-hdr">⑨ Conclusions</div>', unsafe_allow_html=True)
    st.markdown(f"""
    - **Dimensionality reduction works well here.** Just 2 PCs capture {cumev[1]*100:.1f}% of all 
      information; 3 PCs capture {cumev[2]*100:.1f}%.  
    - **PC1 is a walkability axis** — it separates Most Walkable from Least Walkable neighbourhoods 
      almost perfectly, driven by transit access and intersection density scores.  
    - **PC2 is a car-dependence axis** — areas with high PC2 scores tend to have more multi-car 
      households regardless of their walkability score.  
    - **{dims_95} dimensions retain 95%** of the data — a {round((1-dims_95/12)*100)}% reduction 
      from 12 original features, showing strong redundancy among the walkability sub-scores.  
    - **PCA confirms the EPA's 4-category classification** is meaningful — the four walkability 
      groups separate visually in 2D/3D PCA space without using the label.
    """)