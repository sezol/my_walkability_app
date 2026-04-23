import os
"""
Regression Tab — EPA Walkability Index
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
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
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

    features = ["D2A_Ranked", "D2B_Ranked", "D3B_Ranked", "D4A_Ranked",
                "NatWalkInd", "D3B", "D4A_clean", "D2B_E8MIXA",
                "Pct_AO0", "Pct_AO1", "Pct_AO2p"]
    label = "Walkability_Category"

    df_model = df[features + [label]].dropna()
    X = df_model[features]
    y = df_model[label]

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    # Logistic Regression requires scaled features
    scaler = StandardScaler()
    X_tr_scaled = pd.DataFrame(
        scaler.fit_transform(X_tr), columns=features, index=X_tr.index)
    X_te_scaled = pd.DataFrame(
        scaler.transform(X_te), columns=features, index=X_te.index)

    return X_tr, X_te, X_tr_scaled, X_te_scaled, y_tr, y_te, features

@st.cache_data
def run_all_models():
    X_tr, X_te, X_tr_s, X_te_s, y_tr, y_te, features = load_data()

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=42, 
            solver="lbfgs", C=1.0),
        "Gaussian Naive Bayes": GaussianNB(),
        "Decision Tree (Gini, d=5)": DecisionTreeClassifier(
            criterion="gini", max_depth=5, random_state=42),
    }

    results = {}
    for name, clf in models.items():
        # Logistic Regression uses scaled data
        Xtr = X_tr_s if name == "Logistic Regression" else X_tr
        Xte = X_te_s if name == "Logistic Regression" else X_te
        clf.fit(Xtr, y_tr)
        y_pred = clf.predict(Xte)
        results[name] = dict(
            y_pred=y_pred,
            acc=accuracy_score(y_te, y_pred),
            cm=confusion_matrix(y_te, y_pred, labels=CAT_ORDER),
            report=classification_report(y_te, y_pred, output_dict=True),
        )
        # Store coefficients for LR
        if name == "Logistic Regression":
            results[name]["coef"] = clf.coef_
            results[name]["classes"] = clf.classes_

    return results, y_te, features

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

    st.title("Regression")

    # (1) LINEAR REGRESSION
    section("①", "Linear Regression")

    st.markdown(f"""
<div class="info-card" style="border-color:{BLUE};color:#2c3e50">
Linear regression models the relationship between one or more input features and a
<strong>continuous numeric output</strong> by fitting a straight line (or hyperplane)
through the data. The model learns a weight for each feature such that the weighted
sum of inputs best predicts the target value. It minimises the sum of squared
differences between predicted and actual values — known as Ordinary Least Squares (OLS).
Linear regression is best suited for tasks where the output is a real number, such
as predicting house prices, temperature, or a walkability score on a continuous scale.
</div>
""", unsafe_allow_html=True)

    st.latex(r"\hat{y} = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \cdots + \beta_n x_n")

    # ── Visualisation: linear regression line ─────────────────────────
    np.random.seed(42)
    x_demo = np.linspace(1, 20, 120)
    y_demo = 0.8 * x_demo + np.random.normal(0, 2, 120)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.scatter(x_demo, y_demo, color=BLUE, alpha=0.5, s=18, label="Data points")
    m, b = np.polyfit(x_demo, y_demo, 1)
    ax.plot(x_demo, m * x_demo + b, color=RED, lw=2, label=f"Fitted line: y = {m:.2f}x + {b:.2f}")
    ax.set_xlabel("NatWalkInd (walkability score)"); ax.set_ylabel("Predicted continuous output")
    ax.set_title("Linear Regression — Fitting a line to continuous data")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True); plt.close()

    # (2) LOGISTIC REGRESSION
    section("②", "Logistic Regression")

    st.markdown(f"""
<div class="info-card" style="border-color:{GREEN};color:#2c3e50">
Logistic regression is a <strong>classification</strong> model that predicts the
probability that a data point belongs to a particular class. Despite the name, it
does not predict a continuous number — it outputs a value between 0 and 1 by passing
the linear combination of features through the <strong>sigmoid function</strong>.
For multi-class problems (like our four walkability categories), it extends to
<strong>multinomial logistic regression</strong>, which uses a softmax function to
produce a probability for each class simultaneously. The class with the highest
probability is the final prediction.
</div>
""", unsafe_allow_html=True)

    st.latex(r"P(y=1 \mid x) = \sigma(z) = \frac{1}{1 + e^{-z}}, \quad z = \beta_0 + \beta_1 x_1 + \cdots + \beta_n x_n")

    # ── Sigmoid curve ─────────────────────────────────────────────────
    z = np.linspace(-8, 8, 300)
    sig = 1 / (1 + np.exp(-z))

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(z, sig, color=GREEN, lw=2.5, label="Sigmoid: 1 / (1 + e^-z)")
    ax.axhline(0.5, color=ORANGE, ls="--", lw=1.2, label="Decision threshold = 0.5")
    ax.axvline(0,   color="grey",  ls="--", lw=1,   alpha=0.5)
    ax.fill_between(z, sig, 0.5, where=(sig > 0.5), alpha=0.12, color=GREEN, label="Predict class 1")
    ax.fill_between(z, sig, 0.5, where=(sig < 0.5), alpha=0.12, color=RED,   label="Predict class 0")
    ax.set_xlabel("z  (linear combination of features)")
    ax.set_ylabel("P(y = 1 | x)")
    ax.set_title("The Sigmoid Function — squashing any value into (0, 1)")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True); plt.close()

    # (3) SIMILARITIES AND DIFFERENCES
    section("③", "How Are They Similar and Different?")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
<div class="info-card" style="border-color:{BLUE};color:#2c3e50">
<strong>Similarities</strong><br><br>
Both models learn a set of weights (coefficients) for each feature and produce
a prediction as a linear combination of those weights and the input values.
Both assume a linear relationship between the features and the output (or
log-odds in the case of logistic regression). Both are trained by minimising
a loss function, both support regularisation to prevent overfitting, and both
are fast, interpretable, and well-suited to high-dimensional data.
</div>
""", unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
<div class="info-card" style="border-color:{GREEN};color:#2c3e50">
<strong>Differences</strong><br><br>
Linear regression predicts a continuous numeric value and minimises mean squared
error. Logistic regression predicts a class probability (bounded 0 to 1 by the
sigmoid) and minimises cross-entropy loss. Linear regression cannot be used
directly for classification because its outputs are unbounded — a predicted value
of 1.7 or -0.3 has no meaning as a probability. Logistic regression applies the
sigmoid to constrain outputs, making them valid probabilities.
</div>
""", unsafe_allow_html=True)

    # (4) SIGMOID FUNCTION
    section("④", "Does Logistic Regression Use the Sigmoid Function?")

    st.markdown("""
Yes — the sigmoid function is central to logistic regression. The model first
computes a linear score z (the same weighted sum as linear regression), then
passes z through the sigmoid to produce a probability between 0 and 1.
""")

    st.latex(r"\sigma(z) = \frac{1}{1 + e^{-z}}")

    st.markdown("""
The sigmoid has two key properties that make it ideal for classification. First, it
maps any real-valued input to the range (0, 1), so the output can be interpreted as a
probability. Second, it is differentiable everywhere, which allows the model to be
trained using gradient-based optimisation. When z is very large and positive, the
sigmoid outputs a value close to 1 (high confidence in class 1). When z is very
large and negative, the output approaches 0 (high confidence in class 0). The
decision boundary is at z = 0, where the sigmoid outputs exactly 0.5.

For our four-class walkability problem, the multinomial extension replaces the sigmoid
with the **softmax function**, which generalises this idea to produce a probability
distribution across all four classes simultaneously, with all probabilities summing to 1.
""")

    st.latex(r"\text{softmax}(z_k) = \frac{e^{z_k}}{\sum_{j=1}^{K} e^{z_j}}")

    # (5) MAXIMUM LIKELIHOOD
    section("⑤", "Maximum Likelihood and Logistic Regression")

    st.markdown(f"""
<div class="info-card" style="border-color:{PURPLE};color:#2c3e50">
Logistic regression is trained by <strong>Maximum Likelihood Estimation (MLE)</strong>.
The idea is to find the set of weights that makes the observed training labels as probable
as possible under the model. For each training sample, the model outputs a probability
for the true class. MLE multiplies all these probabilities together and finds the weights
that maximise their product — or equivalently, maximise the sum of their logarithms
(the log-likelihood). In practice, we minimise the negative log-likelihood, which is
the <strong>cross-entropy loss</strong>. Gradient descent updates the weights step by
step in the direction that reduces this loss, until the model converges on the weights
that best explain the training data.
</div>
""", unsafe_allow_html=True)

    st.latex(r"\ell(\beta) = \sum_{i=1}^{n} \left[ y_i \log \hat{p}_i + (1 - y_i) \log(1 - \hat{p}_i) \right]")

    st.markdown("""
The connection to our walkability data: each training block group has a known
walkability category. MLE finds the coefficients that maximise the probability of
correctly predicting every block group's category simultaneously. The softmax
extension of this loss is minimised by sklearn's `lbfgs` solver used here.
""")

    explain(
        "Imagine you are guessing what is in a bag by pulling out coloured balls. "
        "Maximum Likelihood says: pick the explanation (the bag contents) that would "
        "make the sequence of balls you actually pulled out the most likely outcome. "
        "For logistic regression, the 'bag' is the set of model weights, and the "
        "'balls' are the correct class labels in the training data.",
        "MLE maximises L(beta) = prod P(y_i | x_i; beta). Taking logs gives the "
        "log-likelihood, which decomposes into a sum. The negative log-likelihood "
        "is the cross-entropy loss: -sum y_i log(p_i). This is convex in beta for "
        "logistic regression, guaranteeing a global minimum reachable by gradient descent."
    )

    # (6) DATA PREP & CODE
    section("⑥", "Data Preparation for Logistic Regression")

    X_tr, X_te, X_tr_s, X_te_s, y_tr, y_te, features = load_data()

    st.markdown("""
Logistic regression is sensitive to feature scale — large-magnitude features dominate
the weight updates during gradient descent. We apply **StandardScaler** (zero mean,
unit variance) before training. The Decision Tree and Naive Bayes models use the
unscaled data for their own runs (as in the previous tabs).
""")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Before scaling — raw features (first 5 rows):**")
        st.dataframe(X_tr.head(), use_container_width=True)
    with col2:
        st.markdown("**After StandardScaler — same rows:**")
        st.dataframe(X_tr_s.head().round(3), use_container_width=True)

    c1, c2, c3, c4 = st.columns(4)
    metric_card(c1, f"{X_tr.shape[0]:,}", "Train Rows",  BLUE)
    metric_card(c2, f"{X_te.shape[0]:,}", "Test Rows",   RED)
    metric_card(c3, f"{X_tr.shape[1]}",   "Features",    GREEN)
    metric_card(c4, "4",                  "Classes",      ORANGE)

    with st.expander("View Full Training Code", expanded=False):
        st.code("""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
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

# Scale for Logistic Regression
scaler   = StandardScaler()
X_tr_s   = scaler.fit_transform(X_tr)
X_te_s   = scaler.transform(X_te)

# Logistic Regression
lr = LogisticRegression(max_iter=1000, 
                         solver="lbfgs", C=1.0, random_state=42)
lr.fit(X_tr_s, y_tr)
print(f"LR  Accuracy: {accuracy_score(y_te, lr.predict(X_te_s)):.4f}")

# Gaussian Naive Bayes (unscaled)
gnb = GaussianNB()
gnb.fit(X_tr, y_tr)
print(f"GNB Accuracy: {accuracy_score(y_te, gnb.predict(X_te)):.4f}")

# Decision Tree (unscaled)
dt = DecisionTreeClassifier(criterion="gini", max_depth=5, random_state=42)
dt.fit(X_tr, y_tr)
print(f"DT  Accuracy: {accuracy_score(y_te, dt.predict(X_te)):.4f}")
""", language="python")

    # (7) RESULTS — ALL THREE MODELS
    section("⑦", "Results — Logistic Regression vs Naive Bayes vs Decision Tree")

    results, y_te, features = run_all_models()
    model_colors = [PURPLE, GREEN, BLUE]

    # Accuracy cards
    cols = st.columns(3)
    for col, (name, res), color in zip(cols, results.items(), model_colors):
        metric_card(col, f"{res['acc']*100:.2f}%", name, color)

    st.markdown("---")

    # ── Logistic Regression coefficients ─────────────────────────────
    st.markdown("#### Logistic Regression — Feature Coefficients")
    st.markdown("""
Unlike Decision Trees and Naive Bayes, Logistic Regression produces a coefficient
for each feature per class. A large positive coefficient means that feature strongly
pushes the model toward predicting that class; a large negative coefficient pushes
it away. This gives us direct insight into which features matter most.
""")

    lr_res = results["Logistic Regression"]
    coef_df = pd.DataFrame(
        lr_res["coef"],
        index=lr_res["classes"],
        columns=features
    ).T

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(features))
    width = 0.2
    class_colors = [RED, ORANGE, GREEN, BLUE]
    for i, (cls, color) in enumerate(zip(lr_res["classes"], class_colors)):
        ax.bar(x + i * width, coef_df[cls], width, label=cls,
               color=color, alpha=0.85, edgecolor="white")
    ax.set_xticks(x + 1.5 * width)
    ax.set_xticklabels(features, rotation=35, ha="right", fontsize=8)
    ax.axhline(0, color="#2c3e50", lw=0.8)
    ax.set_ylabel("Coefficient value")
    ax.set_title("Logistic Regression Coefficients — Feature influence per walkability class")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True); plt.close()

    explain(
        "Each bar shows how strongly a feature pushes the model toward or away from "
        "predicting a walkability category. A tall blue bar for a feature means: when "
        "that feature is high, the model strongly predicts 'Most Walkable'. A tall red "
        "bar going downward means: when that feature is high, the model strongly predicts "
        "away from 'Least Walkable'. Features near zero have little influence.",
        "The coefficient matrix has shape (n_classes, n_features). Each row is the "
        "weight vector for one class in the one-vs-rest decomposition of the multinomial "
        "problem. Standardised inputs allow direct comparison of coefficient magnitudes "
        "across features — larger absolute values indicate stronger predictors."
    )

    st.markdown("---")

    # ── Confusion matrices ────────────────────────────────────────────
    st.markdown("#### Confusion Matrices — All Three Models")

    short_labels = ["Least", "Below Avg", "Above Avg", "Most"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for ax, (name, res), color in zip(axes, results.items(), model_colors):
        cm      = res["cm"]
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        ax.imshow(cm_norm, interpolation="nearest",
                  cmap=sns.light_palette(color, as_cmap=True))
        for i in range(4):
            for j in range(4):
                tc = "white" if cm_norm[i, j] > 0.5 else "#2c3e50"
                ax.text(j, i, f"{cm[i,j]}\n({cm_norm[i,j]*100:.0f}%)",
                        ha="center", va="center", fontsize=8,
                        color=tc, fontweight="bold")
        ax.set_xticks(range(4)); ax.set_yticks(range(4))
        ax.set_xticklabels(short_labels, fontsize=8, rotation=20)
        ax.set_yticklabels(short_labels, fontsize=8)
        ax.set_xlabel("Predicted", fontsize=9); ax.set_ylabel("Actual", fontsize=9)
        ax.set_title(f"{name}\nAccuracy: {res['acc']*100:.2f}%", fontsize=9)

    plt.tight_layout(pad=2)
    st.pyplot(fig, use_container_width=True); plt.close()

    st.markdown("---")

    # ── Per-class F1 ──────────────────────────────────────────────────
    st.markdown("#### Per-Class F1 Scores — All Three Models")

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    for ax, (name, res), color in zip(axes, results.items(), model_colors):
        report = res["report"]
        f1s    = [report.get(c, {}).get("f1-score", 0) for c in CAT_ORDER]
        short  = ["Least", "Below\nAvg", "Above\nAvg", "Most"]
        bars   = ax.bar(short, [v * 100 for v in f1s],
                        color=[color if v > 0.5 else "#d5d8dc" for v in f1s],
                        edgecolor="white", width=0.55)
        for bar, v in zip(bars, f1s):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 1, f"{v*100:.1f}%",
                    ha="center", va="bottom", fontsize=9, color="#2c3e50")
        ax.set_ylim(0, 115); ax.set_ylabel("F1 Score (%)")
        ax.set_title(name, fontsize=9)
        ax.axhline(80, color=ORANGE, ls="--", lw=1, alpha=0.6)
        ax.grid(axis="y", alpha=0.3)

    plt.suptitle("Per-Class F1 Scores", y=1.01, fontsize=11, fontweight="bold")
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True); plt.close()

    st.markdown("---")

    # ── Head-to-head accuracy bar ─────────────────────────────────────
    st.markdown("#### Head-to-Head Accuracy Comparison")

    names = list(results.keys())
    accs  = [results[n]["acc"] * 100 for n in names]

    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.bar(names, accs, color=model_colors, edgecolor="white", width=0.45)
    for bar, v in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.4,
                f"{v:.2f}%", ha="center", va="bottom",
                fontsize=11, fontweight="bold", color="#2c3e50")
    ax.set_ylim(0, 115); ax.set_ylabel("Test Accuracy (%)")
    ax.set_title("Module 3 Model Comparison — Logistic Regression vs NB vs Decision Tree")
    ax.axhline(25, color="grey", ls="--", lw=1, alpha=0.5)
    ax.text(2.55, 26, "Random baseline (25%)", color="grey", fontsize=8)
    ax.set_xticklabels(names, fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True); plt.close()

    best  = max(results, key=lambda k: results[k]["acc"])
    worst = min(results, key=lambda k: results[k]["acc"])
    st.markdown(f"""
<div class="callout green" style="color:#2c3e50">
<strong>Best model: {best} ({results[best]['acc']*100:.2f}%)</strong> —
wins on this dataset due to its ability to learn a linear decision boundary in the
high-dimensional scaled feature space.
</div>
<div class="callout orange" style="color:#2c3e50">
<strong>All models beat the 25% random baseline by a large margin.</strong>
The gap between best and worst is {(results[best]['acc'] - results[worst]['acc'])*100:.1f} percentage
points — meaningful but not dramatic, suggesting the walkability features are
highly informative regardless of which model is used to exploit them.
</div>
""", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════
    # (8) CONCLUSIONS
    # ══════════════════════════════════════════════════════════════════
    section("⑧", "Conclusions")

    lr_acc  = results["Logistic Regression"]["acc"] * 100
    gnb_acc = results["Gaussian Naive Bayes"]["acc"] * 100
    dt_acc  = results["Decision Tree (Gini, d=5)"]["acc"] * 100

    st.markdown(f"""
- **Logistic Regression ({lr_acc:.1f}%)** is the strongest classifier on this dataset
  when features are properly scaled. The multinomial formulation handles all four
  walkability categories in a single model, and the coefficient matrix reveals clearly
  which features drive each class — `NatWalkInd` and the ranked sub-scores carry the
  highest weights, consistent with how the EPA constructed the index.

- **Gaussian Naive Bayes ({gnb_acc:.1f}%)** performs competitively despite its strong
  independence assumption. It benefits from using all 8 continuous features and fitting
  class-specific Gaussian distributions, but the feature correlations it ignores
  (e.g., transit proximity and intersection density are correlated) limit its ceiling.

- **Decision Tree ({dt_acc:.1f}%)** matches or exceeds the other models without requiring
  any feature scaling or probability assumptions. Its axis-aligned splits naturally
  capture the ordinal thresholds in the EPA index, and its structure is the most
  interpretable of the three — a planner can follow the tree path to understand exactly
  why an area was classified as it was.

- **Confusion patterns are consistent across all three models** — adjacent walkability
  categories (*Below Average* and *Above Average*) are hardest to separate because their
  feature distributions overlap at the boundary of the index scale.

- **Key finding:** The four-class walkability classification problem is well-structured
  enough that even simple linear or probabilistic models achieve strong results. The
  dominant predictive features — transit proximity, intersection density, and the
  composite walkability index — confirm the EPA's methodology is both scientifically
  sound and machine-learnable.

- **Recommendation:** For deployment in a planning tool, the Decision Tree is preferred
  for its interpretability. For highest raw accuracy, Logistic Regression is the best
  choice. Naive Bayes is the fastest to train and a strong baseline for quick prototyping.
""")

    st.success("Regression Completed.")
    st.markdown('<div style="margin-bottom:60px"></div>', unsafe_allow_html=True)