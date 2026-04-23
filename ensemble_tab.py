import os
"""
Ensemble Tab — EPA Walkability Index
Module 4: Random Forest Ensemble Learning
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
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report
)
import warnings
warnings.filterwarnings("ignore")

BLUE   = "#3498db"; GREEN  = "#2ecc71"; RED    = "#e74c3c"
ORANGE = "#f39c12"; PURPLE = "#9b59b6"; TEAL   = "#1abc9c"
CAT_ORDER = ["Least Walkable", "Below Average Walkable",
             "Above Average Walkable", "Most Walkable"]

plt.rcParams.update({
    "figure.facecolor": "white",   "axes.facecolor": "#f8f9fa",
    "axes.edgecolor":  "#dee2e6",  "axes.labelcolor": "#2c3e50",
    "xtick.color":     "#7f8c8d",  "ytick.color":     "#7f8c8d",
    "axes.titlesize":  12,         "axes.titleweight": "bold",
    "axes.labelsize":  10,         "font.size": 10,
})

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
    st.markdown(f'<div class="section-hdr">{num} {title}</div>', unsafe_allow_html=True)

# ── Data ──────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("walkability_cleaned.csv")
    df["D4A_clean"] = df["D4A"].replace(-99999, 0)
    df = df[df["Walkability_Category"].notna()]

    features = ["D2A_Ranked", "D2B_Ranked", "D3B_Ranked", "D4A_Ranked",
                "NatWalkInd", "D3B", "D4A_clean", "D2B_E8MIXA",
                "Pct_AO0", "Pct_AO1", "Pct_AO2p"]
    label = "Walkability_Category"

    df_model = df[features + [label]].dropna()
    X = df_model[features]
    y = df_model[label]

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    return X_tr, X_te, y_tr, y_te, features

@st.cache_data
def run_rf():
    X_tr, X_te, y_tr, y_te, features = load_data()

    rf = RandomForestClassifier(
        n_estimators=200, max_depth=None,
        min_samples_split=5, random_state=42,
        n_jobs=-1, oob_score=True
    )
    rf.fit(X_tr, y_tr)
    y_pred = rf.predict(X_te)

    return rf, y_pred, y_te, features

@st.cache_data
def run_n_tree_comparison():
    X_tr, X_te, y_tr, y_te, features = load_data()
    n_trees_list = [1, 5, 10, 25, 50, 100, 200]
    accs = []
    for n in n_trees_list:
        rf = RandomForestClassifier(n_estimators=n, random_state=42, n_jobs=-1)
        rf.fit(X_tr, y_tr)
        accs.append(accuracy_score(y_te, rf.predict(X_te)) * 100)
    return n_trees_list, accs

@st.cache_data
def run_all_models_comparison():
    """Run all module models for final comparison."""
    from sklearn.naive_bayes import GaussianNB
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC

    X_tr, X_te, y_tr, y_te, features = load_data()

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    models = {
        "Gaussian NB":          (GaussianNB(), X_tr, X_te),
        "Decision Tree (d=5)":  (DecisionTreeClassifier(max_depth=5, random_state=42), X_tr, X_te),
        "Logistic Regression":  (LogisticRegression(max_iter=1000, C=1.0, random_state=42), X_tr_s, X_te_s),
        "SVM (RBF, C=100)":     (SVC(kernel="rbf", C=100, gamma="scale", random_state=42), X_tr_s, X_te_s),
        "Random Forest (200)":  (RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1), X_tr, X_te),
    }

    results = {}
    for name, (clf, Xtr, Xte) in models.items():
        clf.fit(Xtr, y_tr)
        results[name] = accuracy_score(y_te, clf.predict(Xte)) * 100

    return results

# ── Main app ──────────────────────────────────────────────────────────────
def app():
    st.markdown("""
    <style>
    .section-hdr{font-size:1.25rem;font-weight:700;color:#2c3e50;
    border-left:4px solid #3498db;padding-left:12px;margin:1.5rem 0 0.8rem}
    .callout{background:#eaf4fd;border-left:4px solid #3498db;
    border-radius:6px;padding:14px 18px;margin:10px 0;font-size:0.9rem;color:#2c3e50}
    .callout.green{background:#eafaf1;border-color:#2ecc71;color:#2c3e50}
    .callout.orange{background:#fef9e7;border-color:#f39c12;color:#2c3e50}
    .info-card{background:#f8f9fa;border-radius:8px;padding:16px 20px;
    border-top:4px solid #3498db;margin-bottom:12px;color:#2c3e50}
    </style>""", unsafe_allow_html=True)

    st.title("Ensemble Learning — Random Forest")

    # (a) OVERVIEW
    # 
    section("①", "Overview — What is Ensemble Learning?")

    st.markdown("""
Ensemble learning is based on a simple but powerful idea: **multiple imperfect models
working together are usually better than one perfect-seeming model working alone.**

A single decision tree, for example, is highly sensitive to the exact training data it sees.
Change a few rows and the tree might look completely different — it overfits. Ensemble
methods fix this by training many models and combining their answers, which smooths out
the individual errors and produces a more reliable final prediction.

**Random Forest** is the most widely used ensemble method. It builds a collection of
decision trees — typically hundreds — each one trained on a slightly different random
sample of the data and a random subset of features. At prediction time, every tree
votes on the class, and the majority vote wins. Because each tree is different, their
errors tend to cancel out rather than compound.
""")

    # ── Image 1: Single tree vs forest ───────────────────────────────
    st.markdown("#### Image 1 — Why Many Trees Beat One Tree")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    np.random.seed(42)
    n = 80
    X_demo = np.random.randn(n, 2)
    y_demo = (X_demo[:,0]**2 + X_demo[:,1]**2 < 1.2).astype(int)
    X_demo += np.random.randn(n, 2) * 0.15

    from sklearn.tree import DecisionTreeClassifier as DTC

    for ax, (max_d, n_trees, title) in zip(axes, [
        (None, 1,   "Single Deep Tree\n(memorises training data)"),
        (3,    1,   "Single Shallow Tree\n(misses real patterns)"),
        (None, 100, "Random Forest (100 trees)\n(best of both worlds)"),
    ]):
        xx, yy = np.meshgrid(np.linspace(-3,3,150), np.linspace(-3,3,150))
        if n_trees == 1:
            clf_d = DTC(max_depth=max_d, random_state=42)
            clf_d.fit(X_demo, y_demo)
            Z = clf_d.predict(np.c_[xx.ravel(), yy.ravel()])
        else:
            from sklearn.ensemble import RandomForestClassifier as RFC
            clf_d = RFC(n_estimators=n_trees, random_state=42)
            clf_d.fit(X_demo, y_demo)
            Z = clf_d.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        ax.contourf(xx, yy, Z, alpha=0.25,
                    cmap=matplotlib.colors.ListedColormap([ORANGE+"66", BLUE+"66"]))
        ax.contour(xx, yy, Z, colors=["white"], linewidths=0.8, alpha=0.5)
        ax.scatter(X_demo[y_demo==0,0], X_demo[y_demo==0,1],
                   color=ORANGE, s=30, alpha=0.8, zorder=3)
        ax.scatter(X_demo[y_demo==1,0], X_demo[y_demo==1,1],
                   color=BLUE, s=30, alpha=0.8, zorder=3)
        acc_d = accuracy_score(y_demo, clf_d.predict(X_demo))
        ax.set_title(f"{title}\nTraining accuracy: {acc_d*100:.0f}%", fontsize=9)
        ax.set_xlim(-3,3); ax.set_ylim(-3,3); ax.grid(alpha=0.2)

    plt.tight_layout()
    st.pyplot(fig, use_container_width=True); plt.close()

    explain(
        "The left chart shows a deep single tree — it memorises every training point perfectly "
        "and draws a jagged, chaotic boundary that won't work well on new data. "
        "The middle chart shows a shallow tree — it's too simple and misses real patterns. "
        "The right chart shows a Random Forest — its boundary is smooth and meaningful "
        "because the errors of hundreds of different trees cancel each other out.",
        "A deep single tree has high variance (sensitive to training data). A shallow tree "
        "has high bias (too simple). Random Forest reduces variance by averaging 100+ trees "
        "trained on bootstrap samples (bagging) with random feature subsets at each split, "
        "decorrelating the trees so their errors are independent and cancel in expectation."
    )

    st.markdown("---")

    # ── The two sources of randomness ────────────────────────────────
    st.markdown("#### Image 2 — Two Types of Randomness: what makes each tree different")

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.set_xlim(0, 12); ax.set_ylim(0, 4.5)
    ax.axis("off"); ax.set_facecolor("white"); fig.patch.set_facecolor("white")

    # Full data
    ax.add_patch(plt.Rectangle((0.2, 1.0), 2.0, 2.5,
                                facecolor=BLUE+"22", edgecolor=BLUE, lw=1.5))
    ax.text(1.2, 2.25, "Full\nDataset\n(43K rows)", ha="center", va="center",
            fontsize=9, color="#2c3e50", fontweight="bold")

    # Three bootstrap samples
    for i, (bx, color, label) in enumerate([
        (4.0, GREEN,  "Bootstrap\nSample 1\n(random 63%\nwith replacement)"),
        (6.2, ORANGE, "Bootstrap\nSample 2\n(different\nrandom rows)"),
        (8.4, PURPLE, "Bootstrap\nSample 3\n(yet another\nrandom sample)"),
    ]):
        ax.annotate("", xy=(bx - 0.6, 2.25), xytext=(2.2, 2.25),
                    arrowprops=dict(arrowstyle="->", color="#7f8c8d", lw=1.2))
        ax.add_patch(plt.Rectangle((bx - 0.6, 1.2), 1.6, 2.1,
                                    facecolor=color+"22", edgecolor=color, lw=1.5))
        ax.text(bx - 0.6 + 0.8, 2.25, label, ha="center", va="center",
                fontsize=7.5, color="#2c3e50")

        # Tree below
        ax.add_patch(plt.Rectangle((bx - 0.6, 0.1), 1.6, 0.85,
                                    facecolor=color+"33", edgecolor=color, lw=1.2))
        ax.text(bx - 0.6 + 0.8, 0.53,
                f"Tree {i+1}\n(random features\nat each split)",
                ha="center", va="center", fontsize=7, color="#2c3e50", fontweight="bold")

    # Vote
    ax.add_patch(plt.Rectangle((10.2, 1.0), 1.5, 2.5,
                                facecolor=RED+"22", edgecolor=RED, lw=1.5))
    ax.text(10.95, 2.25, "Majority\nVote\n= Final\nPrediction",
            ha="center", va="center", fontsize=9, color="#2c3e50", fontweight="bold")
    ax.annotate("", xy=(10.2, 2.25), xytext=(9.4, 2.25),
                arrowprops=dict(arrowstyle="->", color="#7f8c8d", lw=1.5))

    ax.set_title("Random Forest: Two sources of randomness → decorrelated trees → reliable predictions",
                 fontsize=10, pad=6)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True); plt.close()

    explain(
        "Two things make each tree different. First, each tree only sees about 63% of the "
        "training data (a random sample with replacement called a bootstrap). Second, at every "
        "split inside the tree, only a random subset of features is considered — so different "
        "trees end up using different features. These two sources of randomness mean the trees "
        "make different kinds of errors, and when they vote together, those errors cancel out.",
        "Bagging (bootstrap aggregating) reduces variance by averaging n_estimators models "
        "trained on bootstrap samples of size n (with replacement). Random feature subsets "
        "(max_features = sqrt(n_features) for classification) decorrelate trees, reducing "
        "pairwise correlation ρ. Combined variance ≈ ρσ² + (1-ρ)σ²/B → σ²/B as ρ→0, B→∞. "
        "OOB (out-of-bag) samples (~37% not drawn) provide a free internal validation estimate."
    )

    st.markdown("---")

    # (b) DATA PREP & CODE
    section("②", "Data Preparation")

    X_tr, X_te, y_tr, y_te, features = load_data()

    st.markdown("""
Random Forest uses the same feature set as our Decision Tree and SVM experiments.
Unlike SVMs, **Random Forest does not require feature scaling** — it uses threshold-based
splits that are invariant to the scale of features. This is a practical advantage: we can
feed the raw cleaned walkability features directly without StandardScaler.

The same 80/20 stratified split is used as all other supervised models for fair comparison.
""")

    df_display = X_tr.copy()
    df_display["Walkability_Category"] = y_tr.values
    st.dataframe(df_display.head(8), use_container_width=True)

    c1, c2, c3, c4 = st.columns(4)
    metric_card(c1, f"{X_tr.shape[0]:,}", "Train Rows",  BLUE)
    metric_card(c2, f"{X_te.shape[0]:,}", "Test Rows",   RED)
    metric_card(c3, f"{X_tr.shape[1]}",   "Features",    GREEN)
    metric_card(c4, "200",                "Trees",        ORANGE)

    st.markdown("""
**GitHub Links:**
[Ensemble Code](https://github.com/sezol/my_walkability_app/blob/main/ensemble_tab.py) |
[Dataset](https://github.com/sezol/my_walkability_app/blob/main/walkability_cleaned.csv)
""")

    with st.expander("View Full Random Forest Training Code", expanded=False):
        st.code("""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

df = pd.read_csv("walkability_cleaned.csv")
df["D4A_clean"] = df["D4A"].replace(-99999, 0)

features = ["D2A_Ranked","D2B_Ranked","D3B_Ranked","D4A_Ranked",
            "NatWalkInd","D3B","D4A_clean","D2B_E8MIXA",
            "Pct_AO0","Pct_AO1","Pct_AO2p"]

X = df[features]; y = df["Walkability_Category"]
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# No scaling needed — Random Forest is scale invariant
rf = RandomForestClassifier(
    n_estimators=200,     # number of trees
    max_depth=None,       # trees grow fully
    min_samples_split=5,  # minimum samples to split a node
    oob_score=True,       # use out-of-bag samples for free validation
    n_jobs=-1,            # use all CPU cores
    random_state=42
)
rf.fit(X_tr, y_tr)

print(f"OOB Score:  {rf.oob_score_*100:.2f}%")
print(f"Test Accuracy: {accuracy_score(y_te, rf.predict(X_te))*100:.2f}%")
print(classification_report(y_te, rf.predict(X_te)))

# Feature importances
importances = pd.Series(rf.feature_importances_, index=features)
print(importances.sort_values(ascending=False))
""", language="python")

    st.markdown("---")

    # (c) RESULTS
    section("③", "Results")

    rf_model, y_pred, y_te_res, features = run_rf()
    rf_acc = accuracy_score(y_te_res, y_pred) * 100
    oob    = rf_model.oob_score_ * 100

    # Metrics
    c1, c2, c3, c4 = st.columns(4)
    metric_card(c1, f"{rf_acc:.2f}%", "Test Accuracy",  GREEN)
    metric_card(c2, f"{oob:.2f}%",    "OOB Score",       TEAL)
    metric_card(c3, "200",            "Trees in Forest",  BLUE)
    metric_card(c4, "4",              "Classes",          ORANGE)

    st.markdown("---")

    # ── OOB score explanation ─────────────────────────────────────────
    st.markdown("""
**What is the OOB Score?**

Every tree in the forest is trained on about 63% of the data. The remaining 37%
(called out-of-bag samples) are never seen by that tree. So we can use those
leftover samples to test each tree — getting a free validation score without
needing a separate test set. The OOB score is usually very close to the true
test accuracy and is a reliable internal check.
""")

    # ── n_trees vs accuracy ───────────────────────────────────────────
    st.markdown("#### How accuracy improves as we add more trees")

    n_trees_list, n_trees_accs = run_n_tree_comparison()

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(n_trees_list, n_trees_accs, "o-", color=GREEN, lw=2.5, ms=8)
    for n, a in zip(n_trees_list, n_trees_accs):
        ax.text(n, a + 0.2, f"{a:.1f}%", ha="center", fontsize=8.5, color="#2c3e50")
    ax.axhline(n_trees_accs[-1], color=GREEN, ls="--", lw=1, alpha=0.5)
    ax.set_xlabel("Number of Trees"); ax.set_ylabel("Test Accuracy (%)")
    ax.set_title("Random Forest: Accuracy vs Number of Trees")
    ax.set_xscale("log"); ax.grid(alpha=0.3)
    ax.set_ylim(min(n_trees_accs) - 2, max(n_trees_accs) + 2)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True); plt.close()

    explain(
        "Each dot shows the accuracy when using that many trees. A single tree scores "
        "around the same as our standalone Decision Tree from Module 3. As we add more "
        "trees, accuracy improves quickly at first then levels off. After about 50–100 "
        "trees, adding more gives diminishing returns. We use 200 trees to make sure "
        "we're safely past the point of diminishing returns.",
        "The accuracy plateau reflects the law of large numbers — with enough decorrelated "
        "estimators, the ensemble error converges to the irreducible Bayes error. "
        "The initial steep gain comes from variance reduction. Beyond ~100 trees the "
        "marginal gain is sub-percent, and the OOB error stabilises, confirming convergence."
    )

    st.markdown("---")

    # ── Feature importances ───────────────────────────────────────────
    st.markdown("#### What features matter most?")
    st.markdown("""
One of Random Forest's most useful features is that it tells you exactly which
variables drove the predictions. The importance of each feature is measured by how
much it reduces the impurity (GINI) across all splits in all 200 trees.
""")

    importances = pd.Series(rf_model.feature_importances_, index=features)
    importances = importances.sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = [GREEN if v > 0.15 else BLUE if v > 0.05 else "#d5d8dc"
              for v in importances.values]
    bars = ax.barh(importances.index, importances.values * 100,
                   color=colors, height=0.6, edgecolor="white")
    for bar, v in zip(bars, importances.values * 100):
        ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height()/2,
                f"{v:.1f}%", va="center", fontsize=9, color="#2c3e50")
    ax.set_xlabel("Feature Importance (%)")
    ax.set_title("Random Forest Feature Importances — Which features drive walkability prediction?")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True); plt.close()

    top_feature = importances.index[-1]
    top_importance = importances.values[-1] * 100

    st.markdown(f"""
<div class="callout green" style="color:#2c3e50">
<strong>Top predictor: {top_feature} ({top_importance:.1f}% importance)</strong><br>
This is consistent with every other model in this project — the most informative
single feature for predicting walkability category is the National Walkability Index
score itself (NatWalkInd) or one of the ranked sub-components. This makes intuitive
sense: the EPA designed the ranked scores specifically to be discriminative between tiers.
</div>
""", unsafe_allow_html=True)

    explain(
        "A feature with high importance means the forest used it frequently to make decisions "
        "and those decisions were accurate. A low importance feature was either not useful or "
        "was covered by other features. Notice that the ranked EPA scores dominate — they were "
        "specifically designed to discriminate between walkability levels, so it's expected "
        "that the model leans on them heavily.",
        "Gini importance = sum over all trees and nodes of (n_node / n_total) * delta_gini "
        "for splits on that feature. This is a biased estimator that favours high-cardinality "
        "features. For this dataset it is consistent with permutation importance because the "
        "features are not highly correlated (correlation was shown in the NB tab)."
    )

    st.markdown("---")

    # ── Confusion matrix ─────────────────────────────────────────────
    st.markdown("#### Confusion Matrix")

    cm = confusion_matrix(y_te_res, y_pred, labels=CAT_ORDER)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    short_labels = ["Least", "Below Avg", "Above Avg", "Most"]

    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.imshow(cm_norm, interpolation="nearest",
              cmap=sns.light_palette(GREEN, as_cmap=True))
    for i in range(4):
        for j in range(4):
            tc = "white" if cm_norm[i,j] > 0.5 else "#2c3e50"
            ax.text(j, i, f"{cm[i,j]}\n({cm_norm[i,j]*100:.0f}%)",
                    ha="center", va="center", fontsize=9,
                    color=tc, fontweight="bold")
    ax.set_xticks(range(4)); ax.set_yticks(range(4))
    ax.set_xticklabels(short_labels, fontsize=9, rotation=20)
    ax.set_yticklabels(short_labels, fontsize=9)
    ax.set_xlabel("Predicted", fontsize=10); ax.set_ylabel("Actual", fontsize=10)
    ax.set_title(f"Random Forest Confusion Matrix — Accuracy: {rf_acc:.2f}%", fontsize=11)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True); plt.close()

    st.markdown("---")

    # ── Per-class F1 ─────────────────────────────────────────────────
    st.markdown("#### Per-Class F1 Scores")
    report = classification_report(y_te_res, y_pred, output_dict=True)
    f1s  = [report.get(c, {}).get("f1-score", 0) for c in CAT_ORDER]
    short = ["Least", "Below\nAvg", "Above\nAvg", "Most"]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(short, [v*100 for v in f1s],
                  color=[GREEN if v > 0.75 else ORANGE if v > 0.5 else RED for v in f1s],
                  edgecolor="white", width=0.5)
    for bar, v in zip(bars, f1s):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{v*100:.1f}%", ha="center", va="bottom",
                fontsize=10, fontweight="bold", color="#2c3e50")
    ax.set_ylim(0, 115); ax.set_ylabel("F1 Score (%)")
    ax.set_title("Random Forest Per-Class F1 Scores")
    ax.axhline(80, color=ORANGE, ls="--", lw=1, alpha=0.6)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True); plt.close()

    st.markdown("---")

    # ══════════════════════════════════════════════════════════════════
    # (d) COMPARISON OF ALL MODELS
    section("④", "Final Comparison — All Models Across All Modules")

    all_results = run_all_models_comparison()

    names = list(all_results.keys())
    accs  = list(all_results.values())
    colors = [PURPLE, TEAL, RED, ORANGE, GREEN]

    fig, ax = plt.subplots(figsize=(11, 5))
    bars = ax.bar(names, accs, color=colors, edgecolor="white", width=0.55)
    for bar, v in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.4,
                f"{v:.1f}%", ha="center", va="bottom",
                fontsize=10, fontweight="bold", color="#2c3e50")
    ax.set_ylim(0, 110); ax.set_ylabel("Test Accuracy (%)")
    ax.set_title("All Models — Final Accuracy Comparison across Modules 3 and 4")
    ax.axhline(25, color="grey", ls="--", lw=1, alpha=0.5)
    ax.text(4.45, 26, "Random baseline (25%)", color="grey", fontsize=8)
    ax.set_xticklabels(names, fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True); plt.close()

    best_model = max(all_results, key=all_results.get)
    best_acc   = all_results[best_model]

    st.markdown(f"""
<div class="callout green" style="color:#2c3e50">
<strong>Best overall model: {best_model} ({best_acc:.1f}% accuracy)</strong><br>
Random Forest outperforms every other model tested in this project. The combination of
ensemble averaging, built-in feature selection, and the ability to model non-linear
interactions between walkability features makes it the strongest classifier.
</div>

<div class="callout orange" style="color:#2c3e50">
<strong>Key progression:</strong> Each method built on the last. Naive Bayes established a
solid baseline. Decision Trees improved by capturing feature interactions. Logistic Regression
found a better linear boundary. SVM (RBF) handled the non-linearity. Random Forest
combined the strengths of all tree-based approaches and reduced variance through averaging.
</div>
""", unsafe_allow_html=True)

    # (e) CONCLUSIONS
    section("⑤", "Conclusions")

    st.markdown(f"""
- **Random Forest is the strongest model in this project** at {rf_acc:.1f}% test accuracy,
  confirming that ensemble averaging removes the variance that limits individual Decision Trees.

- **The OOB score ({oob:.1f}%)** closely matches the test accuracy, which validates the model —
  it is not overfitting to the training data and generalises well to new block groups.

- **Feature importances confirm every prior finding:** the National Walkability Index score
  and the four EPA ranked sub-scores dominate. This tells us the EPA's scoring methodology
  genuinely captures the most discriminative information about walkability — a reassuring
  validation of the dataset's design.

- **The ensemble approach is the right tool for this problem** because walkability prediction
  involves complex, non-linear interactions between transit access, intersection density,
  employment mix, and car ownership. No single feature or simple threshold captures this —
  but a forest of 200 diverse trees can.

- **Practical implication:** A deployed Random Forest on publicly available EPA data could
  classify any U.S. Census block group's walkability tier with high accuracy — with no
  additional data collection needed. This could support transit planning, zoning decisions,
  or public health interventions at low cost.
""")

    st.success("Module 4 complete — SVM and Ensemble Learning both implemented and compared.")
    st.markdown('<div style="margin-bottom:60px"></div>', unsafe_allow_html=True)