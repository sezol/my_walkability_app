import os
"""
Decision Tree Tab — EPA Walkability Index
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
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
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
        unsafe_allow_html=True)

# ── Data loader ───────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("walkability_cleaned.csv")
    df["D4A_clean"] = df["D4A"].replace(-99999, 0)
    df = df[df["Walkability_Category"].notna()]

    # Feature set (same as MNB + continuous features)
    features = ["D2A_Ranked", "D2B_Ranked", "D3B_Ranked", "D4A_Ranked",
                "NatWalkInd", "D3B", "D4A_clean", "D2B_E8MIXA",
                "Pct_AO0", "Pct_AO1", "Pct_AO2p"]
    label = "Walkability_Category"

    df_model = df[features + [label]].dropna()
    X = df_model[features]
    y = df_model[label]

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    # Raw dataframe for "before" display (use original D4A not cleaned)
    raw_cols = ["D2A_Ranked", "D2B_Ranked", "D3B_Ranked", "D4A_Ranked",
                "NatWalkInd", "D3B", "D4A", "D2B_E8MIXA",
                "Pct_AO0", "Pct_AO1", "Pct_AO2p", "Walkability_Category"]
    raw_df = df[[c for c in raw_cols if c in df.columns]].dropna().head(8)

    return X_tr, X_te, y_tr, y_te, features, raw_df

@st.cache_data
def run_trees():
    X_tr, X_te, y_tr, y_te, features, _ = load_data()

    trees = {
        "Tree 1 — Gini, Max Depth 5": DecisionTreeClassifier(
            criterion="gini", max_depth=5, random_state=42),
        "Tree 2 — Entropy, Max Depth 5": DecisionTreeClassifier(
            criterion="entropy", max_depth=5, random_state=42),
        "Tree 3 — Gini, Max Depth 3": DecisionTreeClassifier(
            criterion="gini", max_depth=3, min_samples_split=50, random_state=42),
    }

    results = {}
    for name, clf in trees.items():
        clf.fit(X_tr, y_tr)
        y_pred = clf.predict(X_te)
        results[name] = dict(
            model=clf,
            y_pred=y_pred,
            acc=accuracy_score(y_te, y_pred),
            cm=confusion_matrix(y_te, y_pred, labels=CAT_ORDER),
            report=classification_report(y_te, y_pred, output_dict=True),
            root=features[clf.tree_.feature[0]],
            n_leaves=clf.get_n_leaves(),
            depth=clf.get_depth(),
        )
    return results, X_tr, X_te, y_tr, y_te, features

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
    .callout.red{background:#fdedec;border-color:#e74c3c;color:#2c3e50}
    .info-card{background:#f8f9fa;border-radius:8px;padding:16px 20px;
    border-top:4px solid #3498db;margin-bottom:12px;color:#2c3e50}
    </style>""", unsafe_allow_html=True)

    st.title("Decision Tree Classification")

    # (1) OVERVIEW
    section("①", "Overview — What is a Decision Tree?")

    st.markdown("""
A Decision Tree is a supervised learning model that learns a series of yes/no questions
about the features in order to classify each data point. Starting from a single root node,
the tree repeatedly splits the data based on the feature that best separates the classes,
branching left or right until it reaches a leaf node that assigns a final prediction.

Decision Trees are popular because they are interpretable — you can trace exactly why the
model made a particular prediction by following the path of questions from root to leaf.
They work for both classification and regression, and require minimal data preprocessing
(no scaling or normalisation needed).
""")

    # ── Diagram 1: Tree structure illustration ────────────────────────
    st.markdown("#### How a Decision Tree is Structured")

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.set_xlim(0, 10); ax.set_ylim(0, 5.5)
    ax.axis("off")

    def node(ax, x, y, text, color, w=2.4, h=0.7):
        box = plt.Rectangle((x - w/2, y - h/2), w, h,
                             facecolor=color, edgecolor="#2c3e50",
                             linewidth=1.5, zorder=3)
        ax.add_patch(box)
        ax.text(x, y, text, ha="center", va="center",
                fontsize=8.5, color="#2c3e50", fontweight="bold", zorder=4,
                wrap=True)

    def arrow(ax, x1, y1, x2, y2, label="", lside=True):
        ax.annotate("", xy=(x2, y2 + 0.35), xytext=(x1, y1 - 0.35),
                    arrowprops=dict(arrowstyle="->", color="#7f8c8d", lw=1.4))
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        if label:
            ax.text(mx + (-0.25 if lside else 0.25), my,
                    label, fontsize=8, color="#7f8c8d", ha="center")

    # Root
    node(ax, 5, 5,    "NatWalkInd <= 10.5?\n(Root Node)", "#d6eaf8")
    # Level 1
    node(ax, 2.5, 3.5, "D4A_Ranked <= 8?\n(Internal Node)", "#d5f5e3")
    node(ax, 7.5, 3.5, "D3B_Ranked <= 14?\n(Internal Node)", "#fdebd0")
    # Level 2 leaves
    node(ax, 1.2, 2,   "Least\nWalkable", "#fadbd8")
    node(ax, 3.8, 2,   "Below Avg\nWalkable", "#fef9e7")
    node(ax, 6.2, 2,   "Above Avg\nWalkable", "#eafaf1")
    node(ax, 8.8, 2,   "Most\nWalkable", "#d6eaf8")

    arrow(ax, 5, 5, 2.5, 3.5, "Yes", lside=True)
    arrow(ax, 5, 5, 7.5, 3.5, "No",  lside=False)
    arrow(ax, 2.5, 3.5, 1.2, 2, "Yes", lside=True)
    arrow(ax, 2.5, 3.5, 3.8, 2, "No",  lside=False)
    arrow(ax, 7.5, 3.5, 6.2, 2, "Yes", lside=True)
    arrow(ax, 7.5, 3.5, 8.8, 2, "No",  lside=False)

    ax.text(5, 0.9, "Leaf nodes assign the final class prediction",
            ha="center", fontsize=9, color="#7f8c8d", style="italic")
    ax.set_facecolor("white"); fig.patch.set_facecolor("white")
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True); plt.close()

    explain(
        "Each box is a question. You start at the top (the root) and answer yes or no. "
        "Each answer sends you down a branch until you reach a coloured leaf box — that "
        "box is the model's prediction. The tree learns which questions to ask, and in "
        "what order, by looking at the training data.",
        "A decision tree recursively partitions the feature space using axis-aligned splits. "
        "At each internal node, the algorithm selects the feature f and threshold t that "
        "maximise the impurity reduction: IG = H(parent) - weighted_avg(H(children)), "
        "where H is the chosen impurity criterion (Gini or Entropy)."
    )

    st.markdown("---")

    # ── GINI / Entropy / IG explanation ──────────────────────────────
    section("②", "Gini Impurity, Entropy, and Information Gain")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"""
<div class="info-card" style="border-color:{BLUE};color:#2c3e50">
<strong>Gini Impurity</strong><br><br>
Measures how often a randomly chosen element from a node would be incorrectly
labelled if it were labelled according to the class distribution at that node.
A Gini of 0 means the node is perfectly pure (all one class).
A Gini of 0.75 means maximum impurity (4 equally likely classes).
<br><br>
<em>Formula:</em> Gini = 1 - sum(p_i^2)
</div>
""", unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
<div class="info-card" style="border-color:{GREEN};color:#2c3e50">
<strong>Entropy and Information Gain</strong><br><br>
Entropy measures the disorder or uncertainty in a node.
Information Gain measures how much a split reduces that disorder.
The algorithm picks the split that produces the highest Information Gain —
meaning the split that creates the purest child nodes.
<br><br>
<em>Formula:</em> Entropy = -sum(p_i * log2(p_i))<br>
<em>Info Gain</em> = Entropy(parent) - weighted avg Entropy(children)
</div>
""", unsafe_allow_html=True)

    # ── Worked example diagram ────────────────────────────────────────
    st.markdown("#### Worked Example — Evaluating a Split on Walkability Data")
    st.markdown("""
Suppose a node contains **100 block groups**: 40 Least Walkable, 30 Below Average,
20 Above Average, 10 Most Walkable. We evaluate splitting on `NatWalkInd <= 10.5`:
""")

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    colors = [RED, ORANGE, GREEN, BLUE]
    short  = ["Least", "Below", "Above", "Most"]

    # Parent node
    parent = [40, 30, 20, 10]
    axes[0].bar(short, parent, color=colors, edgecolor="white", width=0.6)
    axes[0].set_title("Parent Node\n(100 samples)", fontsize=10)
    axes[0].set_ylabel("Count"); axes[0].set_ylim(0, 55)
    for i, v in enumerate(parent):
        axes[0].text(i, v + 0.8, str(v), ha="center", fontsize=9, color="#2c3e50")
    p = np.array(parent) / 100
    gini_p = 1 - np.sum(p**2)
    axes[0].text(1.5, 48, f"Gini = {gini_p:.3f}", ha="center",
                 fontsize=10, color="#2c3e50", fontweight="bold",
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="#f8f9fa", edgecolor="#dee2e6"))

    # Left child (NatWalkInd <= 10.5)
    left = [35, 20, 5, 0]
    axes[1].bar(short, left, color=colors, edgecolor="white", width=0.6)
    axes[1].set_title("Left Child: NatWalkInd <= 10.5\n(60 samples)", fontsize=10)
    axes[1].set_ylabel("Count"); axes[1].set_ylim(0, 55)
    for i, v in enumerate(left):
        axes[1].text(i, v + 0.8, str(v), ha="center", fontsize=9, color="#2c3e50")
    pl = np.array(left) / 60
    gini_l = 1 - np.sum(pl**2)
    axes[1].text(1.5, 48, f"Gini = {gini_l:.3f}", ha="center",
                 fontsize=10, color="#2c3e50", fontweight="bold",
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="#f8f9fa", edgecolor="#dee2e6"))

    # Right child (NatWalkInd > 10.5)
    right = [5, 10, 15, 10]
    axes[2].bar(short, right, color=colors, edgecolor="white", width=0.6)
    axes[2].set_title("Right Child: NatWalkInd > 10.5\n(40 samples)", fontsize=10)
    axes[2].set_ylabel("Count"); axes[2].set_ylim(0, 55)
    for i, v in enumerate(right):
        axes[2].text(i, v + 0.8, str(v), ha="center", fontsize=9, color="#2c3e50")
    pr = np.array(right) / 40
    gini_r = 1 - np.sum(pr**2)
    axes[2].text(1.5, 48, f"Gini = {gini_r:.3f}", ha="center",
                 fontsize=10, color="#2c3e50", fontweight="bold",
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="#f8f9fa", edgecolor="#dee2e6"))

    plt.tight_layout()
    st.pyplot(fig, use_container_width=True); plt.close()

    # Compute and display IG
    weighted_child_gini = (60/100) * gini_l + (40/100) * gini_r
    ig = gini_p - weighted_child_gini

    st.markdown(f"""
<div class="callout" style="color:#2c3e50">
<strong>Information Gain calculation:</strong><br>
Parent Gini = {gini_p:.3f}<br>
Weighted child Gini = (60/100) x {gini_l:.3f} + (40/100) x {gini_r:.3f} = {weighted_child_gini:.3f}<br>
<strong>Information Gain = {gini_p:.3f} - {weighted_child_gini:.3f} = {ig:.3f}</strong><br><br>
A higher Information Gain means the split is better at separating the classes.
The algorithm tests every possible feature and every possible threshold, and picks
the combination with the highest IG as the next split.
</div>
""", unsafe_allow_html=True)

    explain(
        "Imagine you are sorting blocks by colour. Before the split, all 4 colours are mixed "
        "together — very messy (high impurity). After the split at NatWalkInd = 10.5, the left "
        "pile is mostly red (Least Walkable) and the right pile has more of the other colours. "
        "The improvement in tidiness is the Information Gain. The tree always asks: which "
        "question creates the tidiest piles?",
        "Gini = 1 - sum(p_k^2) for k classes. Information Gain = H(S) - sum(|S_v|/|S| * H(S_v)). "
        "Gini tends to isolate the most frequent class into one branch; Entropy is slightly more "
        "balanced due to the logarithmic penalty on small probabilities. In practice both "
        "produce similar trees. sklearn uses Gini by default for speed."
    )

    st.markdown("---")

    # ── Why infinite trees ────────────────────────────────────────────
    section("③", "Why Can Infinitely Many Trees Be Created?")

    st.markdown("""
For any dataset, it is theoretically possible to construct an infinite number of decision trees.
Here is why:

- **Any feature can be the root.** With 11 features in our dataset, there are already 11
  possible root nodes, each producing a completely different tree structure.
- **Any threshold can be used for a split.** For a continuous feature like `NatWalkInd`,
  any real value between its minimum and maximum is a valid split threshold — infinitely many
  choices.
- **Splits can be repeated at any depth.** The same feature can be used at multiple levels
  of the same tree, each with a different threshold.
- **Tree depth is unbounded by default.** Without a `max_depth` constraint, the tree will
  grow until every leaf contains a single sample — creating a perfectly memorised (but
  overfit) model.
- **Different hyperparameters produce different trees.** Changing `min_samples_split`,
  `min_samples_leaf`, `max_features`, or the random seed all produce structurally different trees.
""")

    st.markdown("""
<div class="callout orange" style="color:#2c3e50">
<strong>This is why pruning and depth limits matter.</strong><br>
Without constraints, a decision tree will keep splitting until every training sample is
perfectly classified. This is called overfitting — the model memorises the training data
but fails on new data. Setting max_depth, min_samples_split, or using post-pruning
controls this by limiting the number of possible trees the algorithm can produce.
</div>
""", unsafe_allow_html=True)

    # (4) DATA PREP
    section("④", "Data Preparation")

    X_tr, X_te, y_tr, y_te, features, raw_df = load_data()

    st.markdown("""
**Label:** `Walkability_Category` — four classes: *Least Walkable*, *Below Average Walkable*,
*Above Average Walkable*, *Most Walkable*.

**Features:** 11 variables combining ranked scores and raw continuous EPA measurements.
Decision Trees do not require scaling or normalisation — they split purely on threshold
comparisons, so raw and scaled data produce identical results.
""")

    st.markdown("**Before Transformation — Raw Data from `walkability_cleaned.csv` (first 8 rows):**")
    st.markdown("""
This is the full cleaned dataset as loaded. All columns are numeric. No additional
cleaning is needed for Decision Trees since they are scale-invariant, but the `D4A`
sentinel value (−99,999) is replaced with 0 before training.
""")
    st.dataframe(raw_df, use_container_width=True)

    st.markdown("---")
    st.markdown("**After Transformation — Feature matrix fed to the Decision Tree (first 8 rows of training set):**")

    # Show sample dataframe
    df_display = X_tr.copy()
    df_display["Walkability_Category"] = y_tr.values
    st.dataframe(df_display.head(8), use_container_width=True)
    st.markdown(f"**Training shape:** {X_tr.shape[0]:,} rows × {X_tr.shape[1]} features")

    # Train/test split visual
    fig, ax = plt.subplots(figsize=(10, 1.6))
    ax.barh(0, 80, color=BLUE, height=0.5, label="Training Set (80%)")
    ax.barh(0, 20, left=80, color=RED, height=0.5, label="Test Set (20%)")
    ax.set_xlim(0, 100); ax.set_yticks([])
    ax.set_xlabel("Percentage of Data")
    ax.set_title("Train / Test Split — Stratified by Walkability Category", pad=10)
    ax.text(40, 0, "Training Set (80%)", ha="center", va="center",
            color="white", fontweight="bold", fontsize=11)
    ax.text(90, 0, "Test (20%)", ha="center", va="center",
            color="white", fontweight="bold", fontsize=10)
    ax.legend(loc="upper right", fontsize=9); ax.grid(False)
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True); plt.close()

    c1, c2, c3, c4 = st.columns(4)
    metric_card(c1, f"{X_tr.shape[0]:,}", "Train Rows",    BLUE)
    metric_card(c2, f"{X_te.shape[0]:,}", "Test Rows",     RED)
    metric_card(c3, f"{X_tr.shape[1]}",   "Features",      GREEN)
    metric_card(c4, "4",                  "Classes",        ORANGE)

    explain(
        "The 80% training set is what the tree learns from. The 20% test set is held back "
        "completely and only used at the end to measure accuracy on unseen data. Stratified "
        "splitting ensures all four walkability categories appear in both sets in proportion "
        "to how common they are in the full dataset.",
        "Stratified train_test_split(stratify=y, test_size=0.2, random_state=42) preserves "
        "class priors in both subsets. Decision trees are invariant to monotonic feature "
        "transformations so no StandardScaler is applied — splits are threshold-based and "
        "not affected by scale."
    )

    # (5) CODE
    section("⑤", "Code")

    with st.expander("View Full Training Code", expanded=False):
        st.code("""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

df = pd.read_csv("walkability_cleaned.csv")
df["D4A_clean"] = df["D4A"].replace(-99999, 0)

features = ["D2A_Ranked", "D2B_Ranked", "D3B_Ranked", "D4A_Ranked",
            "NatWalkInd", "D3B", "D4A_clean", "D2B_E8MIXA",
            "Pct_AO0", "Pct_AO1", "Pct_AO2p"]

X = df[features]
y = df["Walkability_Category"]

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# Tree 1 — Gini, max depth 5
dt1 = DecisionTreeClassifier(criterion="gini", max_depth=5, random_state=42)
dt1.fit(X_tr, y_tr)
print(f"Tree 1 Accuracy : {accuracy_score(y_te, dt1.predict(X_te)):.4f}")
print(f"Tree 1 Root Node: {features[dt1.tree_.feature[0]]}")

# Tree 2 — Entropy, max depth 5
dt2 = DecisionTreeClassifier(criterion="entropy", max_depth=5, random_state=42)
dt2.fit(X_tr, y_tr)
print(f"Tree 2 Accuracy : {accuracy_score(y_te, dt2.predict(X_te)):.4f}")
print(f"Tree 2 Root Node: {features[dt2.tree_.feature[0]]}")

# Tree 3 — Gini, max depth 3, min_samples_split=50 (more conservative)
dt3 = DecisionTreeClassifier(criterion="gini", max_depth=3,
                               min_samples_split=50, random_state=42)
dt3.fit(X_tr, y_tr)
print(f"Tree 3 Accuracy : {accuracy_score(y_te, dt3.predict(X_te)):.4f}")
print(f"Tree 3 Root Node: {features[dt3.tree_.feature[0]]}")
""", language="python")

    # (6) RESULTS
    section("⑥", "Results")

    results, X_tr, X_te, y_tr, y_te, features = run_trees()

    # Accuracy cards
    cols = st.columns(3)
    tree_colors = [BLUE, GREEN, ORANGE]
    for col, (name, res), color in zip(cols, results.items(), tree_colors):
        short_name = name.split("—")[1].strip()
        metric_card(col, f"{res['acc']*100:.2f}%", short_name, color)

    st.markdown("---")

    # ── Tree summaries ────────────────────────────────────────────────
    st.markdown("#### Tree Structure Summary")

    summary_data = []
    for name, res in results.items():
        summary_data.append({
            "Tree": name,
            "Root Node": res["root"],
            "Max Depth Reached": res["depth"],
            "Number of Leaves": res["n_leaves"],
            "Test Accuracy": f"{res['acc']*100:.2f}%",
        })
    st.dataframe(pd.DataFrame(summary_data), use_container_width=True, hide_index=True)

    explain(
        "The root node is the very first question the tree asks — the single feature that "
        "best splits the entire training set. Different trees have different root nodes "
        "because they use different criteria (Gini vs Entropy) or different depth limits, "
        "which affects which split looks best at the top of the tree. The number of leaves "
        "tells you how many distinct prediction buckets the tree has created.",
        "The root node is the feature with the highest impurity reduction on the full "
        "training set. Tree 1 and Tree 2 use different impurity criteria (Gini vs Entropy), "
        "which can yield different root features when two features have similar but not equal "
        "impurity reduction scores. Tree 3's min_samples_split=50 changes the effective "
        "splits available, potentially altering the root as well."
    )

    st.markdown("---")

    # ── Decision Tree Visualisations ──────────────────────────────────
    st.markdown("#### Decision Tree Visualisations (Top 3 Levels)")

    short_features = ["D2A_R", "D2B_R", "D3B_R", "D4A_R",
                      "WalkInd", "D3B", "D4A", "MixA",
                      "AO0", "AO1", "AO2p"]
    short_classes  = ["Least", "BelowAvg", "AboveAvg", "Most"]

    for (name, res), color in zip(results.items(), tree_colors):
        st.markdown(f"**{name}** — Root: `{res['root']}` | Depth: {res['depth']} | Leaves: {res['n_leaves']} | Accuracy: {res['acc']*100:.2f}%")
        fig, ax = plt.subplots(figsize=(18, 6))
        plot_tree(
            res["model"],
            feature_names=short_features,
            class_names=short_classes,
            filled=True,
            rounded=True,
            max_depth=3,
            fontsize=8,
            ax=ax,
            impurity=True,
            proportion=False,
        )
        ax.set_title(f"{name}", fontsize=11, pad=10)
        fig.patch.set_facecolor("white")
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True); plt.close()
        st.markdown("---")

    # ── Confusion Matrices ────────────────────────────────────────────
    st.markdown("#### Confusion Matrices")

    short_labels = ["Least", "Below Avg", "Above Avg", "Most"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, (name, res), color in zip(axes, results.items(), tree_colors):
        cm = res["cm"]
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

        ax.imshow(cm_norm, interpolation="nearest",
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
        short_name = name.split("—")[1].strip()
        ax.set_title(f"{short_name}\nAccuracy: {res['acc']*100:.2f}%", fontsize=10)

    plt.tight_layout(pad=2)
    st.pyplot(fig, use_container_width=True); plt.close()

    explain(
        "Each grid compares what the model predicted (columns) to what was actually true "
        "(rows). Bright cells on the diagonal mean the model got it right. Off-diagonal "
        "cells are mistakes. Notice that misclassifications mostly occur between neighbouring "
        "categories — a 'Below Average' area being predicted as 'Least Walkable' is a "
        "smaller error than predicting it as 'Most Walkable'.",
        "Normalised confusion matrices show per-class recall (TP_k / sum(row_k)). "
        "Off-diagonal density between adjacent categories reflects the ordinal structure "
        "of the label — boundary block groups share overlapping feature distributions. "
        "Extreme categories (Least/Most Walkable) have higher recall because their "
        "feature distributions are more distant from other classes."
    )

    st.markdown("---")

    # ── Per-class F1 ──────────────────────────────────────────────────
    st.markdown("#### Per-Class F1 Scores")

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    for ax, (name, res), color in zip(axes, results.items(), tree_colors):
        report = res["report"]
        f1s = [report.get(c, {}).get("f1-score", 0) for c in CAT_ORDER]
        short = ["Least", "Below\nAvg", "Above\nAvg", "Most"]
        bars = ax.bar(short, [v * 100 for v in f1s],
                      color=[color if v > 0.5 else "#d5d8dc" for v in f1s],
                      edgecolor="white", width=0.55)
        for bar, v in zip(bars, f1s):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 1, f"{v*100:.1f}%",
                    ha="center", va="bottom", fontsize=9, color="#2c3e50")
        ax.set_ylim(0, 115); ax.set_ylabel("F1 Score (%)")
        short_name = name.split("—")[1].strip()
        ax.set_title(short_name, fontsize=10)
        ax.axhline(80, color=ORANGE, ls="--", lw=1, alpha=0.6)
        ax.grid(axis="y", alpha=0.3)

    plt.suptitle("Per-Class F1 Scores across the three Decision Trees",
                 y=1.01, fontsize=11, fontweight="bold")
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True); plt.close()

    # ── Overall accuracy comparison ───────────────────────────────────
    st.markdown("#### Overall Accuracy Comparison")

    names = list(results.keys())
    accs  = [results[n]["acc"] * 100 for n in names]
    short_names = [n.split("—")[1].strip() for n in names]

    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.bar(short_names, accs, color=tree_colors, edgecolor="white", width=0.45)
    for bar, v in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.4,
                f"{v:.2f}%", ha="center", va="bottom",
                fontsize=11, fontweight="bold", color="#2c3e50")
    ax.set_ylim(0, 115); ax.set_ylabel("Test Accuracy (%)")
    ax.set_title("Decision Tree — Overall Accuracy by Configuration")
    ax.axhline(25, color="grey", ls="--", lw=1, alpha=0.5)
    ax.text(2.55, 26, "Random baseline (25%)", color="grey", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True); plt.close()

    best = max(results, key=lambda k: results[k]["acc"])
    st.markdown(f"""
<div class="callout green" style="color:#2c3e50">
<strong>Best performer: {best} ({results[best]['acc']*100:.2f}% accuracy)</strong> —
all three trees significantly outperform the 25% random baseline.
</div>

<div class="callout orange" style="color:#2c3e50">
<strong>Gini vs Entropy:</strong> Both criteria use the same features and depth limit,
so their accuracies are close. The key difference is computational — Gini avoids a
logarithm calculation and is slightly faster. Entropy can produce marginally different
splits when two features have very similar impurity reductions.<br><br>
<strong>Depth 3 vs Depth 5:</strong> The shallower Tree 3 (max_depth=3) trades some
accuracy for interpretability and reduced overfitting risk — fewer leaves means the
model generalises more broadly.
</div>
""", unsafe_allow_html=True)

    # (7) CONCLUSIONS
    
    section("⑦", "Conclusions")

    t1_acc = results["Tree 1 — Gini, Max Depth 5"]["acc"] * 100
    t2_acc = results["Tree 2 — Entropy, Max Depth 5"]["acc"] * 100
    t3_acc = results["Tree 3 — Gini, Max Depth 3"]["acc"] * 100

    st.markdown(f"""
- **Decision Trees are highly effective on walkability data**, with all three configurations
  achieving well above the 25% random baseline. The best model reaches {max(t1_acc, t2_acc, t3_acc):.1f}% accuracy.

- **The root node in all trees is `NatWalkInd`** (or a closely related ranked score) — confirming
  that the composite National Walkability Index is the single most discriminative feature for
  separating the four walkability categories, which aligns with how the EPA designed the index.

- **Gini and Entropy produce nearly identical results** ({t1_acc:.1f}% vs {t2_acc:.1f}%) when
  depth and other hyperparameters are equal. This is consistent with the literature — both
  criteria measure impurity and typically agree on the best split.

- **Depth 3 tree ({t3_acc:.1f}%) is only slightly less accurate than depth 5**, suggesting the
  top 3 levels of the tree already capture most of the predictive signal. This is valuable for
  interpretability — a 3-level tree can be read and explained to a non-technical audience.

- **Confusion patterns** mirror those from Naive Bayes — adjacent categories (*Below Average*
  and *Above Average*) are hardest to separate because their walkability index scores overlap
  at the boundary. Extreme categories are classified most reliably.

- **Takeaway for urban planning:** A shallow decision tree with just NatWalkInd and the four
  ranked sub-scores can reliably classify any US Census block group into its walkability tier.
  This kind of simple, interpretable model could be used by city planners to flag areas for
  infrastructure investment with high transparency and without requiring complex ML expertise.
""")

    st.success("Decision Tree analysis complete. Proceed to the Regression tab.")
    st.markdown('<div style="margin-bottom:60px"></div>', unsafe_allow_html=True)