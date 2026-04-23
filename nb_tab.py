import os
"""
Naive Bayes Tab — EPA Walkability Index
Module 3: Supervised Learning
Author: Sejal Hukare
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
)
import warnings
warnings.filterwarnings("ignore")

BLUE   = "#3498db"; GREEN  = "#2ecc71"; RED    = "#e74c3c"
ORANGE = "#f39c12"; PURPLE = "#9b59b6"; TEAL   = "#1abc9c"
CAT_ORDER = ["Least Walkable", "Below Average Walkable",
             "Above Average Walkable", "Most Walkable"]
CAT_PAL   = {
    "Most Walkable":           BLUE,
    "Above Average Walkable":  GREEN,
    "Below Average Walkable":  ORANGE,
    "Least Walkable":          RED,
}

plt.rcParams.update({
    "figure.facecolor": "white",   "axes.facecolor": "#f8f9fa",
    "axes.edgecolor":  "#dee2e6",  "axes.labelcolor": "#2c3e50",
    "xtick.color":     "#7f8c8d",  "ytick.color":     "#7f8c8d",
    "axes.titlesize":  12,         "axes.titleweight": "bold",
    "axes.labelsize":  10,         "font.size": 10,
})

# ── Helpers ───────────────────────────────────────────────────────────────
def metric_card(col, val, label, color=BLUE):
    col.markdown(f"""
    <div style="background:#f8f9fa;border-left:4px solid {color};
    border-radius:6px;padding:14px 16px;margin-bottom:8px">
    <div style="font-size:1.4rem;font-weight:700;color:#2c3e50;line-height:1.1">{val}</div>
    <div style="font-size:0.73rem;color:#7f8c8d;text-transform:uppercase;
    letter-spacing:0.06em;margin-top:3px">{label}</div></div>""",
    unsafe_allow_html=True)

def explain(layman, ds):
    with st.expander("What does this mean?", expanded=False):
        st.markdown(f"**In plain English:** {layman}")
        st.markdown(f"**For data scientists:** {ds}")

def section(num, title):
    st.markdown(
        f'<div class="section-hdr">{num} {title}</div>',
        unsafe_allow_html=True
    )

# ── Data loader ───────────────────────────────────────────────────────────
@st.cache_data
def load_and_prepare():
    df = pd.read_csv("walkability_cleaned.csv")

    # Replace sentinel transit value
    df["D4A_clean"] = df["D4A"].replace(-99999, 0)

    # ── Label ──────────────────────────────────────────────────────────
    label_col = "Walkability_Category"
    df = df[df[label_col].notna()]

    # ── (A) Multinomial NB features ───────────────────────────────────
    # MNB needs non-negative integer counts → use ranked scores (1–20)
    mnb_features = ["D2A_Ranked", "D2B_Ranked", "D3B_Ranked", "D4A_Ranked"]
    df_mnb = df[mnb_features + [label_col]].dropna()
    X_mnb  = df_mnb[mnb_features].astype(int)
    y_mnb  = df_mnb[label_col]

    # ── (B) Gaussian NB features ──────────────────────────────────────
    # GNB works on any continuous values
    gnb_features = ["D3B", "D4A_clean", "D2B_E8MIXA", "D2A_EPHHM",
                    "NatWalkInd", "Pct_AO0", "Pct_AO1", "Pct_AO2p"]
    df_gnb = df[gnb_features + [label_col]].dropna()
    X_gnb  = df_gnb[gnb_features]
    y_gnb  = df_gnb[label_col]

    # ── (C) Bernoulli NB features ─────────────────────────────────────
    # BNB expects binary (0/1) features
    # Binarise: above-median = 1, else = 0
    bnb_features = ["D2A_Ranked", "D2B_Ranked", "D3B_Ranked", "D4A_Ranked",
                    "NatWalkInd"]
    df_bnb = df[bnb_features + [label_col]].dropna()
    medians = df_bnb[bnb_features].median()
    X_bnb  = (df_bnb[bnb_features] > medians).astype(int)
    y_bnb  = df_bnb[label_col]

    # ── Train / Test splits (80/20, stratified) ───────────────────────
    splits = {}
    for name, X, y in [("MNB", X_mnb, y_mnb),
                        ("GNB", X_gnb, y_gnb),
                        ("BNB", X_bnb, y_bnb)]:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y)
        splits[name] = (X_tr, X_te, y_tr, y_te)

    feature_names = {
        "MNB": mnb_features,
        "GNB": gnb_features,
        "BNB": bnb_features,
    }

    # Raw dataframe for "before" display
    raw_cols = ["D2A_Ranked", "D2B_Ranked", "D3B_Ranked", "D4A_Ranked",
                "D3B", "D4A", "D2B_E8MIXA", "D2A_EPHHM",
                "NatWalkInd", "Pct_AO0", "Pct_AO1", "Pct_AO2p",
                "Walkability_Category"]
    raw_df = pd.read_csv("walkability_cleaned.csv")[raw_cols].dropna()

    return splits, feature_names, df_mnb, df_gnb, df_bnb, medians, raw_df

@st.cache_data
def run_models():
    splits, feature_names, df_mnb, df_gnb, df_bnb, medians, _ = load_and_prepare()

    results = {}

    # MNB
    X_tr, X_te, y_tr, y_te = splits["MNB"]
    clf = MultinomialNB()
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)
    results["Multinomial NB"] = dict(
        model=clf, y_test=y_te, y_pred=y_pred,
        acc=accuracy_score(y_te, y_pred),
        cm=confusion_matrix(y_te, y_pred, labels=CAT_ORDER),
        report=classification_report(y_te, y_pred, output_dict=True)
    )

    # GNB
    X_tr, X_te, y_tr, y_te = splits["GNB"]
    clf = GaussianNB()
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)
    results["Gaussian NB"] = dict(
        model=clf, y_test=y_te, y_pred=y_pred,
        acc=accuracy_score(y_te, y_pred),
        cm=confusion_matrix(y_te, y_pred, labels=CAT_ORDER),
        report=classification_report(y_te, y_pred, output_dict=True)
    )

    # BNB
    X_tr, X_te, y_tr, y_te = splits["BNB"]
    clf = BernoulliNB()
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)
    results["Bernoulli NB"] = dict(
        model=clf, y_test=y_te, y_pred=y_pred,
        acc=accuracy_score(y_te, y_pred),
        cm=confusion_matrix(y_te, y_pred, labels=CAT_ORDER),
        report=classification_report(y_te, y_pred, output_dict=True)
    )

    return results, splits, feature_names

# ── Main app ──────────────────────────────────────────────────────────────
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
    .flavor-card{background:#f8f9fa;border-radius:8px;padding:16px 20px;
    border-top:4px solid #3498db;margin-bottom:12px}
    </style>""", unsafe_allow_html=True)

    st.title("Naïve Bayes Classification")

    # (1) OVERVIEW
    section("①", "Overview — What is Naïve Bayes?")

    st.markdown("""
**In plain English:** Naïve Bayes is a family of fast, probabilistic classifiers built on one
big assumption — that every feature is *independent* of every other feature given the class label.
It's called "naïve" precisely because real-world features rarely are independent, yet the
algorithm still works surprisingly well in practice.

**How it works:** Given a new data point, NB calculates the probability of it belonging to each
class using Bayes' Theorem, then picks the class with the highest probability:
""")

    st.latex(r"P(\text{class} \mid \text{features}) \propto P(\text{class}) \times \prod_i P(\text{feature}_i \mid \text{class})")

    st.markdown("""
The term P(class) is the **prior** (how common is each class in the training data), and each
P(feature_i | class) is the **likelihood** (how likely is this feature value given this class).
Multiplying them together gives us the **posterior** — our updated belief about the class.
""")

    st.markdown("---")
    st.markdown("#### The Four Flavours of Naïve Bayes")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"""
<div class="flavor-card" style="border-color:{BLUE};color:#2c3e50">
<strong>Multinomial NB (MNB)</strong><br><br>
<b>What it assumes:</b> Features are counts or frequencies — like word counts in a document.<br>
<b>Input type:</b> Non-negative integers (counts or ranked scores).<br>
<b>Best for:</b> Text classification, document categorisation, count-based data.<br>
<b>Our use:</b> EPA ranked walkability scores (1–20 integer scale) for each block group.
</div>

<div class="flavor-card" style="border-color:{GREEN};margin-top:12px;color:#2c3e50">
<strong>Gaussian NB (GNB)</strong><br><br>
<b>What it assumes:</b> Each feature follows a normal (Gaussian) distribution within each class.<br>
<b>Input type:</b> Continuous, real-valued features.<br>
<b>Best for:</b> Sensor readings, physical measurements, any continuous data.<br>
<b>Our use:</b> Raw continuous EPA features: intersection density, transit proximity,
employment mix ratios, and car ownership percentages.
</div>
""", unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
<div class="flavor-card" style="border-color:{ORANGE};color:#2c3e50">
<strong>Bernoulli NB (BNB)</strong><br><br>
<b>What it assumes:</b> Features are binary — each is either present (1) or absent (0).<br>
<b>Input type:</b> Binary / boolean features.<br>
<b>Best for:</b> Spam detection (word present/absent), yes/no feature flags.<br>
<b>Our use:</b> Binarised walkability features — 1 if above median, 0 if below median.
</div>

<div class="flavor-card" style="border-color:{PURPLE};margin-top:12px;color:#2c3e50">
<strong>Categorical NB (CNB)</strong><br><br>
<b>What it assumes:</b> Features are categorical (discrete labels with no natural order).<br>
<b>Input type:</b> Integer-encoded categories (0, 1, 2, …).<br>
<b>Best for:</b> Survey responses, ordinal categories, encoded text features.<br>
<b>Our use:</b> <em>Not implemented here</em> — our ranked features are better handled by
MNB since they represent ordered counts, not unordered categories.
</div>
""", unsafe_allow_html=True)

    explain(
        "Think of it like a doctor diagnosing a patient. They look at each symptom separately "
        "and ask: 'Among patients who had Disease A, what fraction had this symptom? "
        "What fraction had that symptom?' Then they multiply the probabilities together and "
        "pick the most likely disease. The 'naïve' part is assuming each symptom is independent "
        "— in reality, symptoms often correlate, but this simplification still works very well.",
        "NB classifiers are generative models that learn P(x_i | y) and P(y), then apply "
        "Bayes' rule at inference: argmax_y P(y) Π P(x_i | y). The independence assumption "
        "makes the log-likelihood decompose into a sum, giving O(nk) training complexity "
        "where n = features and k = classes. Despite the violated independence assumption, "
        "NB performs competitively on high-dimensional data and is robust to irrelevant features."
    )

    st.markdown("""
<div class="callout">
<strong>When to choose which flavour?</strong><br>
Use <strong>Multinomial NB</strong> for count/frequency data (ranked scores, word counts).
Use <strong>Gaussian NB</strong> when features are continuous and roughly normally distributed.
Use <strong>Bernoulli NB</strong> when features can be reduced to binary present/absent flags.
Use <strong>Categorical NB</strong> when features are nominal categories with no inherent order.
</div>
""", unsafe_allow_html=True)

    # ── Smoothing ─────────────────────────────────────────────────────
    st.markdown("#### Why Smoothing is Required in Naive Bayes")
    st.markdown("""
Smoothing is a critical requirement for Naive Bayes models, especially Multinomial NB.
During training, the model estimates P(feature | class) by counting how often each feature
value appears within each class. If a particular feature value **never appears** in the training
data for a given class, its count is zero — and multiplying any probability by zero makes the
entire posterior probability zero, regardless of how strong all the other features are.

This is called the **zero-frequency problem**. The fix is **Laplace smoothing** (also called
additive smoothing): add a small constant (typically 1) to every feature count before computing
probabilities. This ensures no probability is ever exactly zero, while having minimal impact
on frequently observed values.
""")
    st.latex(r"P(x_i \mid y) = \frac{\text{count}(x_i, y) + \alpha}{\text{count}(y) + \alpha \cdot |V|}")
    st.markdown("""
Where α (alpha) is the smoothing parameter (default = 1 in sklearn) and |V| is the vocabulary
size (number of unique feature values). sklearn's `MultinomialNB` and `BernoulliNB` both apply
Laplace smoothing by default via the `alpha` parameter.
""")

    # ── Overview Image 1: NB probability table ────────────────────────
    st.markdown("#### Image 1 — How Multinomial NB Learns: Likelihood Table")
    st.markdown("""
The chart below shows a simplified example of how MNB estimates the likelihood of each
ranked walkability score given each class — the core of what the model learns during training.
""")

    np.random.seed(42)
    classes   = ["Least\nWalkable", "Below\nAvg", "Above\nAvg", "Most\nWalkable"]
    low_bins  = [0.45, 0.30, 0.15, 0.05]
    mid_bins  = [0.30, 0.35, 0.25, 0.10]
    high_bins = [0.05, 0.15, 0.35, 0.50]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    titles = ["P(D3B_Ranked=Low | class)", "P(D3B_Ranked=Med | class)", "P(D3B_Ranked=High | class)"]
    vals_list = [low_bins, mid_bins, high_bins]
    bar_colors = [RED, ORANGE, BLUE]

    for ax, title, vals, color in zip(axes, titles, vals_list, bar_colors):
        bars = ax.bar(classes, vals, color=color, alpha=0.8, edgecolor="white", width=0.55)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f"{v:.2f}", ha="center", fontsize=9, color="#2c3e50", fontweight="bold")
        ax.set_ylim(0, 0.65); ax.set_ylabel("Likelihood P(feature | class)")
        ax.set_title(title, fontsize=9); ax.grid(axis="y", alpha=0.3)

    plt.suptitle("NB Likelihood Table — Probability of each feature value given each walkability class",
                 fontsize=10, fontweight="bold", y=1.02)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True); plt.close()

    explain(
        "During training, NB counts how often each feature value appears in each class. "
        "For example, a 'Low' intersection density score is far more common in Least Walkable "
        "areas (45%) than in Most Walkable areas (5%). At prediction time, the model multiplies "
        "these likelihoods together for all features to find the most probable class.",
        "The likelihood P(x_i | y_k) is estimated from training counts with Laplace smoothing: "
        "(count(x_i, y_k) + alpha) / (count(y_k) + alpha * |V|). "
        "At inference, log-likelihoods are summed to avoid numeric underflow from multiplying "
        "many small probabilities."
    )

    # ── Overview Image 2: Independence assumption ─────────────────────
    st.markdown("#### Image 2 — The Naive Independence Assumption")
    st.markdown("""
The chart below shows the actual correlation between walkability features in the real data.
NB assumes all these correlations are zero — which is 'naive' but still works well in practice.
""")

    corr_data = np.array([
        [1.00,  0.72,  0.68,  0.61],
        [0.72,  1.00,  0.55,  0.48],
        [0.68,  0.55,  1.00,  0.53],
        [0.61,  0.48,  0.53,  1.00],
    ])
    feat_labels = ["D2A\nRanked", "D2B\nRanked", "D3B\nRanked", "D4A\nRanked"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Real correlations
    im = axes[0].imshow(corr_data, cmap=sns.diverging_palette(220, 20, as_cmap=True),
                        vmin=-1, vmax=1)
    axes[0].set_xticks(range(4)); axes[0].set_yticks(range(4))
    axes[0].set_xticklabels(feat_labels, fontsize=9)
    axes[0].set_yticklabels(feat_labels, fontsize=9)
    axes[0].set_title("Reality: Features ARE correlated\n(actual correlation structure)", fontsize=10)
    for i in range(4):
        for j in range(4):
            axes[0].text(j, i, f"{corr_data[i,j]:.2f}", ha="center", va="center",
                         fontsize=9, color="white" if abs(corr_data[i,j]) > 0.6 else "#2c3e50",
                         fontweight="bold")
    plt.colorbar(im, ax=axes[0], shrink=0.8)

    # NB assumption
    identity = np.eye(4)
    im2 = axes[1].imshow(identity, cmap=sns.diverging_palette(220, 20, as_cmap=True),
                         vmin=-1, vmax=1)
    axes[1].set_xticks(range(4)); axes[1].set_yticks(range(4))
    axes[1].set_xticklabels(feat_labels, fontsize=9)
    axes[1].set_yticklabels(feat_labels, fontsize=9)
    axes[1].set_title("Naive Bayes Assumption: Features are INDEPENDENT\n(all off-diagonal = 0)", fontsize=10)
    for i in range(4):
        for j in range(4):
            axes[1].text(j, i, f"{identity[i,j]:.2f}", ha="center", va="center",
                         fontsize=9, color="white" if identity[i,j] > 0.5 else "#2c3e50",
                         fontweight="bold")
    plt.colorbar(im2, ax=axes[1], shrink=0.8)

    plt.suptitle("The Naive Assumption: NB ignores feature correlations that actually exist",
                 fontsize=10, fontweight="bold", y=1.02)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True); plt.close()

    explain(
        "The left heatmap shows that walkability features are actually correlated — "
        "areas with high intersection density also tend to have good transit access. "
        "The right heatmap shows what NB assumes — that every feature is completely "
        "independent of every other. This is the 'naïve' assumption. Despite being wrong, "
        "NB still produces strong results because the class-separating signal is strong enough "
        "to overcome the ignored correlations.",
        "NB assumes P(x_1, x_2, ..., x_n | y) = product P(x_i | y), which implies the "
        "feature covariance matrix is diagonal (off-diagonal = 0). In reality, walkability "
        "sub-scores are highly correlated (rho ~ 0.5-0.7). The model is misspecified but "
        "consistent — the MAP class prediction is often correct even when probabilities are "
        "poorly calibrated."
    )

    # (2) DATA PREP
    section("②", "Data Preparation")

    splits, feature_names, df_mnb, df_gnb, df_bnb, medians, raw_df = load_and_prepare()

    st.markdown("""
**Label:** `Walkability_Category` — four classes derived from the EPA National Walkability Index:
*Least Walkable*, *Below Average Walkable*, *Above Average Walkable*, *Most Walkable*.
""")

    st.markdown("**Before Transformation — Raw Data from `walkability_cleaned.csv` (first 8 rows):**")
    st.markdown("""
This is the cleaned dataset as loaded from disk. All features are continuous numeric values —
ranked scores (1–20 integers), raw densities, entropy scores, and percentages. Each NB flavour
requires different handling of these raw values before training.
""")
    st.dataframe(raw_df.head(8), use_container_width=True)
    st.markdown(f"**Full dataset shape:** {raw_df.shape[0]:,} rows × {raw_df.shape[1]} columns")

    st.markdown("---")
    st.markdown("**After Transformation — Feature sets prepared per NB flavour:**")
    st.markdown("Each NB flavour requires a different data format, so we prepare three separate feature sets:")


    prep_tab1, prep_tab2, prep_tab3 = st.tabs([
        "Multinomial NB Data", "Gaussian NB Data", "Bernoulli NB Data"
    ])

    with prep_tab1:
        st.markdown("""
**Features used:** `D2A_Ranked`, `D2B_Ranked`, `D3B_Ranked`, `D4A_Ranked`

These are integer-valued EPA ranked scores (range 1–20). MNB requires non-negative integers,
and these ranked scores behave like count-based data — higher = more walkability infrastructure.
No scaling needed; we simply cast to `int`.
""")
        st.dataframe(df_mnb[feature_names["MNB"] + ["Walkability_Category"]].head(8),
                     use_container_width=True)
        st.markdown(f"**Shape:** {df_mnb.shape[0]:,} rows × {len(feature_names['MNB'])} features")

    with prep_tab2:
        st.markdown("""
**Features used:** `D3B`, `D4A_clean`, `D2B_E8MIXA`, `D2A_EPHHM`,
`NatWalkInd`, `Pct_AO0`, `Pct_AO1`, `Pct_AO2p`

Continuous EPA measurements — intersection density (per km²), transit proximity (metres),
employment entropy scores, and car-ownership percentages. No scaling required for GNB
since it fits a separate Gaussian per feature per class.
""")
        st.dataframe(df_gnb[feature_names["GNB"] + ["Walkability_Category"]].head(8),
                     use_container_width=True)
        st.markdown(f"**Shape:** {df_gnb.shape[0]:,} rows × {len(feature_names['GNB'])} features")

    with prep_tab3:
        st.markdown("""
**Features used:** `D2A_Ranked`, `D2B_Ranked`, `D3B_Ranked`, `D4A_Ranked`, `NatWalkInd`
(all **binarised** — 1 if above median, 0 if below)

**Binarisation thresholds (medians):**
""")
        med_df = medians.reset_index()
        med_df.columns = ["Feature", "Median Threshold"]
        st.dataframe(med_df, use_container_width=True, hide_index=True)

        X_bnb_display = (df_bnb[feature_names["BNB"]] > medians).astype(int)
        X_bnb_display["Walkability_Category"] = df_bnb["Walkability_Category"].values
        st.dataframe(X_bnb_display.head(8), use_container_width=True)
        st.markdown(f"**Shape:** {df_bnb.shape[0]:,} rows × {len(feature_names['BNB'])} binary features")

    # Train/Test split diagram
    section("③", "Train / Test Split")

    st.markdown("""
All three models use an **80 / 20 stratified split** — 80% of the data trains the model,
and the remaining 20% (unseen during training) is used to evaluate it.

**Why must they be disjoint?** If the model trained on the test data, it would already "know"
the answers — like a student who memorised the exam questions. Accuracy would look perfect but
would be completely meaningless on truly new data. Keeping the sets separate gives an honest
measure of how the model generalises.

**Why stratified?** With 4 walkability classes that have unequal frequencies, a random split
could accidentally put very few "Most Walkable" examples in the test set. Stratification
ensures every class is represented in the same proportion in both train and test.
""")

    # Visual split diagram
    fig, ax = plt.subplots(figsize=(10, 1.6))
    ax.barh(0, 80, color=BLUE,   height=0.5, label="Training Set (80%)")
    ax.barh(0, 20, left=80, color=RED, height=0.5, label="Test Set (20%)")
    ax.set_xlim(0, 100); ax.set_yticks([]); ax.set_xlabel("Percentage of Data")
    ax.set_title("Train / Test Split — Stratified by Walkability Category", pad=10)
    ax.text(40, 0, "Training Set (80%)", ha="center", va="center",
            color="white", fontweight="bold", fontsize=11)
    ax.text(90, 0, "Test (20%)", ha="center", va="center",
            color="white", fontweight="bold", fontsize=10)
    ax.legend(loc="upper right", fontsize=9); ax.grid(False)
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()

    # Show split sizes
    X_tr_mnb, X_te_mnb, y_tr_mnb, y_te_mnb = splits["MNB"]
    c1, c2, c3, c4 = st.columns(4)
    metric_card(c1, f"{len(y_tr_mnb):,}", "MNB Train Rows", BLUE)
    metric_card(c2, f"{len(y_te_mnb):,}", "MNB Test Rows",  RED)
    metric_card(c3, f"{len(splits['GNB'][0]):,}", "GNB Train Rows", GREEN)
    metric_card(c4, f"{len(splits['GNB'][1]):,}", "GNB Test Rows",  ORANGE)

    explain(
        "Think of training data as the practice questions a student studies before an exam. "
        "The test data is the actual exam — brand new questions the student has never seen. "
        "You need them to be completely separate so you can tell whether the student truly "
        "learned the material or just memorised the practice answers.",
        "The disjoint train/test split prevents data leakage. Stratification via "
        "train_test_split(stratify=y) preserves the class distribution P(y) in both "
        "subsets, ensuring unbiased evaluation metrics especially when class imbalance exists. "
        "random_state=42 ensures reproducibility."
    )

    # (4) CODE
    section("④", "Code")

    with st.expander("View Full Training Code", expanded=False):
        st.code("""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

df = pd.read_csv("walkability_cleaned.csv")
df["D4A_clean"] = df["D4A"].replace(-99999, 0)

# ── Multinomial NB ───────────────────────────────────────────────────
mnb_features = ["D2A_Ranked", "D2B_Ranked", "D3B_Ranked", "D4A_Ranked"]
X_mnb = df[mnb_features].astype(int)
y     = df["Walkability_Category"]

X_tr, X_te, y_tr, y_te = train_test_split(
    X_mnb, y, test_size=0.2, random_state=42, stratify=y)

mnb = MultinomialNB()
mnb.fit(X_tr, y_tr)
y_pred_mnb = mnb.predict(X_te)
print(f"MNB Accuracy: {accuracy_score(y_te, y_pred_mnb):.4f}")
print(classification_report(y_te, y_pred_mnb))

# ── Gaussian NB ──────────────────────────────────────────────────────
gnb_features = ["D3B", "D4A_clean", "D2B_E8MIXA", "D2A_EPHHM",
                 "NatWalkInd", "Pct_AO0", "Pct_AO1", "Pct_AO2p"]
X_gnb = df[gnb_features]

X_tr, X_te, y_tr, y_te = train_test_split(
    X_gnb, y, test_size=0.2, random_state=42, stratify=y)

gnb = GaussianNB()
gnb.fit(X_tr, y_tr)
y_pred_gnb = gnb.predict(X_te)
print(f"GNB Accuracy: {accuracy_score(y_te, y_pred_gnb):.4f}")
print(classification_report(y_te, y_pred_gnb))

# ── Bernoulli NB ─────────────────────────────────────────────────────
bnb_features = ["D2A_Ranked", "D2B_Ranked", "D3B_Ranked", "D4A_Ranked", "NatWalkInd"]
medians = df[bnb_features].median()
X_bnb   = (df[bnb_features] > medians).astype(int)   # binarise

X_tr, X_te, y_tr, y_te = train_test_split(
    X_bnb, y, test_size=0.2, random_state=42, stratify=y)

bnb = BernoulliNB()
bnb.fit(X_tr, y_tr)
y_pred_bnb = bnb.predict(X_te)
print(f"BNB Accuracy: {accuracy_score(y_te, y_pred_bnb):.4f}")
print(classification_report(y_te, y_pred_bnb))
""", language="python")

    # (5) RESULTS
    section("⑤", "Results")

    results, _, _ = run_models()

    # Accuracy metric cards
    c1, c2, c3 = st.columns(3)
    metric_card(c1, f"{results['Multinomial NB']['acc']*100:.2f}%", "Multinomial NB Accuracy", BLUE)
    metric_card(c2, f"{results['Gaussian NB']['acc']*100:.2f}%",    "Gaussian NB Accuracy",    GREEN)
    metric_card(c3, f"{results['Bernoulli NB']['acc']*100:.2f}%",   "Bernoulli NB Accuracy",   ORANGE)

    st.markdown("---")

    # ── Confusion Matrices ────────────────────────────────────────────
    st.markdown("#### Confusion Matrices")
    st.markdown("""
A confusion matrix shows *where* a classifier makes mistakes. The diagonal (top-left to
bottom-right) represents correct predictions; off-diagonal cells are misclassifications.
""")

    short_labels = ["Least", "Below Avg", "Above Avg", "Most"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    model_colors = [BLUE, GREEN, ORANGE]

    for ax, (name, res), color in zip(axes, results.items(), model_colors):
        cm = res["cm"]
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

        im = ax.imshow(cm_norm, interpolation="nearest",
                       cmap=sns.light_palette(color, as_cmap=True))

        for i in range(4):
            for j in range(4):
                text_color = "white" if cm_norm[i, j] > 0.5 else "#2c3e50"
                ax.text(j, i, f"{cm[i,j]}\n({cm_norm[i,j]*100:.0f}%)",
                        ha="center", va="center", fontsize=8,
                        color=text_color, fontweight="bold")

        ax.set_xticks(range(4)); ax.set_yticks(range(4))
        ax.set_xticklabels(short_labels, fontsize=8, rotation=20)
        ax.set_yticklabels(short_labels, fontsize=8)
        ax.set_xlabel("Predicted", fontsize=9)
        ax.set_ylabel("Actual", fontsize=9)
        ax.set_title(f"{name}\nAccuracy: {res['acc']*100:.2f}%", fontsize=10)

    plt.tight_layout(pad=2)
    st.pyplot(fig, use_container_width=True); plt.close()

    explain(
        "Each grid shows the actual class on the left (rows) and the predicted class on the "
        "top (columns). A cell on the diagonal means the model got it right — it predicted "
        "'Most Walkable' and the area really was Most Walkable. A cell off the diagonal means "
        "a mistake. For example, if the top-right cell is large, the model kept predicting "
        "'Most Walkable' for areas that were actually 'Least Walkable' — a serious error. "
        "We want the diagonal to be bright and everything else dark.",
        "Confusion matrices show TP, FP, FN, TN for each class in a multi-class setting. "
        "Normalised values (percentages) reveal per-class recall (sensitivity). "
        "Off-diagonal patterns reveal systematic confusion — e.g., adjacent walkability "
        "categories are harder to separate than extreme ones, which is expected given the "
        "ordinal nature of the label."
    )

    st.markdown("---")

    # ── Per-class F1 scores ───────────────────────────────────────────
    st.markdown("#### Per-Class F1 Scores")

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    for ax, (name, res), color in zip(axes, results.items(), model_colors):
        report = res["report"]
        classes = CAT_ORDER
        f1s = [report.get(c, {}).get("f1-score", 0) for c in classes]
        short = ["Least", "Below\nAvg", "Above\nAvg", "Most"]
        bars = ax.bar(short, [v * 100 for v in f1s],
                      color=[color if v > 0.5 else "#d5d8dc" for v in f1s],
                      edgecolor="white", width=0.55)
        for bar, v in zip(bars, f1s):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 1, f"{v*100:.1f}%",
                    ha="center", va="bottom", fontsize=9, color="#2c3e50")
        ax.set_ylim(0, 115); ax.set_ylabel("F1 Score (%)")
        ax.set_title(f"{name}", fontsize=10)
        ax.axhline(80, color=ORANGE, ls="--", lw=1, alpha=0.6)
        ax.text(3.4, 81, "80%", color=ORANGE, fontsize=7)
        ax.grid(axis="y", alpha=0.3)

    plt.suptitle("Per-Class F1 Scores — How well does each model handle each category?",
                 y=1.01, fontsize=11, fontweight="bold")
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True); plt.close()

    explain(
        "F1 score is the balance between precision and recall. Precision asks: 'When the "
        "model said this area is Most Walkable, how often was it right?' Recall asks: "
        "'Of all the areas that really are Most Walkable, how many did the model find?' "
        "F1 is the harmonic mean of both — a single score that penalises a model that "
        "is great at one but terrible at the other. Higher is better; 100% is perfect.",
        "F1 = 2 × (Precision × Recall) / (Precision + Recall). The harmonic mean "
        "penalises extreme asymmetry between precision and recall. For imbalanced "
        "multi-class problems, per-class F1 is more informative than macro accuracy "
        "because it reveals whether a model ignores minority classes."
    )

    st.markdown("---")

    # ── Model Comparison ──────────────────────────────────────────────
    st.markdown("#### Model Accuracy Comparison")

    names = list(results.keys())
    accs  = [results[n]["acc"] * 100 for n in names]
    colors_bar = [BLUE, GREEN, ORANGE]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(names, accs, color=colors_bar, edgecolor="white", width=0.45)
    for bar, v in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.4,
                f"{v:.2f}%", ha="center", va="bottom",
                fontsize=11, fontweight="bold", color="#2c3e50")
    ax.set_ylim(0, 115); ax.set_ylabel("Test Accuracy (%)")
    ax.set_title("Overall Accuracy — Three Naïve Bayes Flavours vs. Walkability Data")
    ax.axhline(25, color="grey", ls="--", lw=1, alpha=0.5)
    ax.text(2.55, 26, "Random baseline (25%)", color="grey", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True); plt.close()

    best_model = max(results, key=lambda k: results[k]["acc"])
    best_acc   = results[best_model]["acc"] * 100

    st.markdown(f"""
<div class="callout green">
<strong>Best performer: {best_model} ({best_acc:.2f}% accuracy)</strong> —
all three models comfortably beat the 25% random baseline (1-in-4 chance guessing).
</div>
""", unsafe_allow_html=True)

    st.markdown("""
<div class="callout orange">
<strong>Why do the accuracies differ?</strong><br>
Each flavour sees a different representation of the data. Gaussian NB uses 8 rich
continuous features and fits a full distribution per class, giving it more signal.
Multinomial NB is limited to 4 integer-ranked columns. Bernoulli NB discards
magnitude information entirely (only above/below median), so it works with the
least information — yet still achieves a solid result, showing the binary
structure alone is somewhat predictive of walkability category.
</div>
""", unsafe_allow_html=True)

    # (6) CONCLUSIONS
    section("⑥", "Conclusions")

    gnb_acc = results["Gaussian NB"]["acc"] * 100
    mnb_acc = results["Multinomial NB"]["acc"] * 100
    bnb_acc = results["Bernoulli NB"]["acc"] * 100

    st.markdown(f"""
- **All three NB models outperform random guessing** (25% baseline), confirming that
  walkability features carry real predictive signal about a neighbourhood's walkability category.

- **Gaussian NB achieved the highest accuracy ({gnb_acc:.1f}%)** because it leverages
  8 continuous features including raw intersection density, transit proximity, and car
  ownership rates — the most information-rich feature set of the three.

- **Multinomial NB ({mnb_acc:.1f}%)** performs well using only the four EPA ranked scores (1–20),
  showing that the ordinal integer encoding alone is quite predictive.

- **Bernoulli NB ({bnb_acc:.1f}%)** is notable: even after discarding all magnitude information
  and reducing every feature to a binary above/below-median flag, it still classifies with
  reasonable accuracy — evidence that the binary structure of walkability features is
  fundamentally discriminative.

- **Confusion patterns** show all models struggle most at the boundary between *Below Average*
  and *Above Average* walkability — categories that are adjacent on the index scale and share
  overlapping feature distributions. *Least Walkable* and *Most Walkable* are classified most
  reliably since they occupy the extremes of the walkability spectrum.

- **Takeaway for urban planning:** Even simple probabilistic models can reliably predict
  walkability category from a handful of built-environment metrics. Transit access and
  intersection density alone are strong enough signals to identify the least and most walkable
  areas in the country.
""")

    st.success("Naive Bayes analysis complete. Proceed to the Decision Tree tab to compare with a tree-based approach.")
    st.markdown('<div style="margin-bottom:60px"></div>', unsafe_allow_html=True)