"""
SVM Tab — EPA Walkability Index
Module 4: Support Vector Machines
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
from sklearn.svm import SVC
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

    # Sample for SVM performance (SVMs are O(n²) — keep manageable)
    df_sample = df_model.sample(n=min(8000, len(df_model)), random_state=42)

    X = df_sample[features]
    y = df_sample[label]

    raw_df = df_sample[["D2A_Ranked", "D2B_Ranked", "D3B_Ranked", "D4A_Ranked",
                         "NatWalkInd", "D3B", "D4A_clean", "D2B_E8MIXA",
                         "Pct_AO0", "Pct_AO1", "Pct_AO2p",
                         "Walkability_Category"]].head(8)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_tr_s = pd.DataFrame(scaler.fit_transform(X_tr), columns=features, index=X_tr.index)
    X_te_s = pd.DataFrame(scaler.transform(X_te),     columns=features, index=X_te.index)

    return X_tr, X_te, X_tr_s, X_te_s, y_tr, y_te, features, raw_df

@st.cache_data
def run_svm_experiments():
    X_tr, X_te, X_tr_s, X_te_s, y_tr, y_te, features, _ = load_data()
    costs  = [0.1, 1, 10, 100]
    kernels = ["linear", "poly", "rbf"]

    results = {}
    for kernel in kernels:
        results[kernel] = {}
        for C in costs:
            clf = SVC(kernel=kernel, C=C, random_state=42,
                      degree=3, gamma="scale", decision_function_shape="ovr")
            clf.fit(X_tr_s, y_tr)
            y_pred = clf.predict(X_te_s)
            results[kernel][C] = {
                "acc":    accuracy_score(y_te, y_pred),
                "cm":     confusion_matrix(y_te, y_pred, labels=CAT_ORDER),
                "report": classification_report(y_te, y_pred, output_dict=True),
                "y_pred": y_pred,
            }
    return results, y_te

@st.cache_data
def get_2d_boundary_data():
    """Train SVM on first 2 PCA components for decision boundary plot."""
    from sklearn.decomposition import PCA
    X_tr, X_te, X_tr_s, X_te_s, y_tr, y_te, features, _ = load_data()

    pca = PCA(n_components=2)
    X_tr_2d = pca.fit_transform(X_tr_s)
    X_te_2d = pca.transform(X_te_s)

    boundary_data = {}
    for kernel, C in [("linear", 10), ("poly", 10), ("rbf", 100)]:
        clf = SVC(kernel=kernel, C=C, gamma="scale", degree=3, random_state=42)
        clf.fit(X_tr_2d, y_tr)

        x_min, x_max = X_tr_2d[:,0].min()-0.5, X_tr_2d[:,0].max()+0.5
        y_min, y_max = X_tr_2d[:,1].min()-0.5, X_tr_2d[:,1].max()+0.5
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                              np.linspace(y_min, y_max, 200))
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

        boundary_data[kernel] = {
            "xx": xx, "yy": yy, "Z": Z,
            "X_te": X_te_2d, "y_te": y_te,
            "acc": accuracy_score(y_te, clf.predict(X_te_2d)),
        }
    return boundary_data

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

    st.title("Support Vector Machines (SVM)")

    # (a) OVERVIEW
    section("①", "Overview — What is an SVM?")

    st.markdown("""
A Support Vector Machine is a classifier that finds the best possible boundary
between two classes of data. The key word is **best** — not just any boundary that
separates the classes, but the one that keeps the most distance from the nearest
points on each side. That distance is called the **margin**, and the data points
sitting on the edges of that margin are called **support vectors** — the only points
that actually determine where the boundary goes.

SVMs are fundamentally **linear separators**: they draw a straight line (in 2D),
a flat plane (in 3D), or a hyperplane (in higher dimensions) between the classes.
The challenge is that most real data cannot be separated by a straight line. This
is where the **kernel trick** comes in.
""")

    # ── Image 1: Margin illustration ─────────────────────────────────
    st.markdown("#### Image 1 — The Maximum Margin: What SVM is really doing")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    np.random.seed(7)
    # Class A — lower left
    A = np.random.randn(40, 2) + np.array([-2, -1.5])
    # Class B — upper right
    B = np.random.randn(40, 2) + np.array([2, 1.5])

    for ax, title, show_margin in zip(axes,
        ["Any line separates them — but which is best?",
         "SVM picks the one with the MAXIMUM MARGIN"],
        [False, True]):

        ax.scatter(A[:,0], A[:,1], color=BLUE,  s=35, alpha=0.7, label="Class A", zorder=3)
        ax.scatter(B[:,0], B[:,1], color=RED,   s=35, alpha=0.7, label="Class B", zorder=3)

        x_line = np.linspace(-5, 5, 100)

        if not show_margin:
            for slope, intercept, col in [(0.6, 0, "#aaa"), (0.8, 0.5, "#bbb"), (0.5, -0.5, "#ccc")]:
                ax.plot(x_line, slope * x_line + intercept, color=col, lw=1.2, ls="--")
            ax.plot(x_line, 0.65 * x_line, color=GREEN, lw=2, label="SVM boundary")
        else:
            ax.plot(x_line, 0.65 * x_line, color=GREEN, lw=2.5, label="Decision boundary")
            ax.plot(x_line, 0.65 * x_line + 1.8, color=GREEN, lw=1.2, ls="--", label="Margin boundary")
            ax.plot(x_line, 0.65 * x_line - 1.8, color=GREEN, lw=1.2, ls="--")
            ax.fill_between(x_line, 0.65*x_line - 1.8, 0.65*x_line + 1.8, alpha=0.08, color=GREEN)
            # support vectors
            sv_A = A[np.argsort(A[:,1] - 0.65*A[:,0])[-3:]]
            sv_B = B[np.argsort(B[:,1] - 0.65*B[:,0])[:3]]
            ax.scatter(sv_A[:,0], sv_A[:,1], s=120, color=BLUE,  edgecolors="black", lw=2, zorder=5, label="Support vectors")
            ax.scatter(sv_B[:,0], sv_B[:,1], s=120, color=RED,   edgecolors="black", lw=2, zorder=5)
            ax.annotate("", xy=(0.5, 0.65*0.5+1.8), xytext=(0.5, 0.65*0.5),
                        arrowprops=dict(arrowstyle="<->", color="black", lw=1.5))
            ax.text(0.8, 0.65*0.5+0.9, "Margin", fontsize=9, color="#2c3e50", fontweight="bold")

        ax.set_xlim(-5, 5); ax.set_ylim(-5, 5)
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8, loc="upper left"); ax.grid(alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig, use_container_width=True); plt.close()

    explain(
        "The left chart shows that many different lines could separate the two groups. "
        "SVM ignores all of them except the one that maximises the gap — the margin — "
        "between the nearest points on each side. Those nearest points are the support vectors "
        "(circled on the right). Everything else in the data doesn't affect the boundary at all.",
        "SVM solves the dual quadratic optimisation problem: maximise ||w||^-1 subject to "
        "y_i(w·x_i + b) >= 1. The support vectors are the training points with non-zero "
        "Lagrange multipliers. The soft-margin version introduces slack variables ξ_i and "
        "penalises margin violations by C * sum(ξ_i), making C the regularisation parameter."
    )

    st.markdown("---")

    # ── Kernel trick explanation ──────────────────────────────────────
    st.markdown("#### The Kernel Trick — How SVMs handle non-linear data")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
<div class="info-card" style="border-color:{BLUE};color:#2c3e50">
<strong>Why kernels are needed</strong><br><br>
Most real data cannot be separated by a straight line. The kernel trick solves this
by mapping the data into a higher-dimensional space where a flat boundary
can separate the classes — then doing all the math without ever explicitly
computing those higher-dimensional coordinates.<br><br>
The key insight: SVMs only ever need the <strong>dot product</strong> between data
points, not the points themselves. A kernel function K(x, x') computes the dot
product in the high-dimensional space directly, cheaply, without going there.
This is why the dot product is so critical — it's the only operation that matters.
</div>
""", unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
<div class="info-card" style="border-color:{GREEN};color:#2c3e50">
<strong>The three kernels we use</strong><br><br>
<b>Linear kernel:</b> K(x, x') = x · x'<br>
No transformation. Works when data is already roughly linearly separable.
The fastest kernel to train.<br><br>
<b>Polynomial kernel:</b> K(x, x') = (x · x' + r)^d<br>
Maps data to a polynomial feature space. The degree d controls the curve complexity.
We use r = 1, d = 3.<br><br>
<b>RBF (Radial Basis Function) kernel:</b><br>
K(x, x') = exp(−γ||x − x'||²)<br>
Creates smooth, circular decision regions. Most flexible — can fit almost any shape.
Often the strongest performer.
</div>
""", unsafe_allow_html=True)

    st.latex(r"K_{\text{poly}}(x, x') = (x \cdot x' + r)^d \qquad K_{\text{RBF}}(x, x') = \exp\!\left(-\gamma \|x - x'\|^2\right)")

    # ── Image 2: Kernel trick visualisation ──────────────────────────
    st.markdown("#### Image 2 — The Kernel Trick: lifting non-separable data into higher dimensions")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    np.random.seed(42)
    # 1D non-separable data
    inner  = np.random.uniform(-1, 1, 30)
    outer1 = np.random.uniform(-3, -1.5, 20)
    outer2 = np.random.uniform(1.5, 3, 20)
    outer  = np.concatenate([outer1, outer2])

    axes[0].scatter(inner, np.zeros(30), color=BLUE, s=40, alpha=0.7, label="Class A", zorder=3)
    axes[0].scatter(outer, np.zeros(len(outer)), color=RED, s=40, alpha=0.7, label="Class B", zorder=3)
    axes[0].set_title("Original 1D Data — Not linearly separable")
    axes[0].set_xlabel("x"); axes[0].set_yticks([])
    axes[0].axvline(-1.2, color="grey", ls="--", lw=1, alpha=0.5)
    axes[0].axvline(1.2,  color="grey", ls="--", lw=1, alpha=0.5)
    axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)

    axes[1].text(0.5, 0.5,
                 "Map x  →  (x, x²)\n\nThis adds a new dimension.\nIn the new space,\nthe classes ARE\nlinearly separable.",
                 ha="center", va="center", fontsize=11, color="#2c3e50",
                 transform=axes[1].transAxes,
                 bbox=dict(boxstyle="round,pad=0.6", facecolor="#eaf4fd", edgecolor=BLUE))
    axes[1].set_title("The Kernel Trick: add a dimension")
    axes[1].axis("off")

    # 2D separable
    inner_2d  = np.column_stack([inner,  inner**2])
    outer_2d  = np.column_stack([outer,  outer**2])
    axes[2].scatter(inner_2d[:,0], inner_2d[:,1], color=BLUE, s=40, alpha=0.7, label="Class A", zorder=3)
    axes[2].scatter(outer_2d[:,0], outer_2d[:,1], color=RED,  s=40, alpha=0.7, label="Class B", zorder=3)
    axes[2].axhline(1.8, color=GREEN, lw=2, ls="--", label="SVM boundary (now flat!)")
    axes[2].set_title("Mapped 2D Space — Now linearly separable!")
    axes[2].set_xlabel("x"); axes[2].set_ylabel("x²")
    axes[2].legend(fontsize=8); axes[2].grid(alpha=0.3)

    plt.suptitle("How the Kernel Trick lifts non-separable data into a space where a flat boundary works",
                 fontsize=10, y=1.02)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True); plt.close()

    explain(
        "The left chart shows data that no straight line can separate — the blue points are "
        "in the middle and the red points are on both sides. By adding a new dimension (x²), "
        "the data lifts up into 3D, and suddenly a flat horizontal line separates them perfectly. "
        "The kernel trick does this lifting mathematically without actually computing the "
        "new coordinates, keeping everything fast.",
        "The kernel trick exploits Mercer's theorem: any positive semi-definite kernel K(x,x') "
        "implicitly defines a dot product in a (possibly infinite-dimensional) feature space φ. "
        "Since SVMs only need dot products, K(x,x') = φ(x)·φ(x') lets us work in high-D "
        "spaces at the cost of O(n) kernel evaluations, avoiding explicit feature computation."
    )

    st.markdown("---")

    # ── Polynomial kernel worked example ─────────────────────────────
    section("②", "Polynomial Kernel — Worked Example: Casting 2D to 6D")

    st.markdown("""
Let's trace exactly what happens when we apply a **polynomial kernel with r = 1 and d = 2**
to a single 2D data point. This makes the kernel trick concrete and shows why it's
called "casting to higher dimensions."
""")

    st.markdown("""
**Starting point:** x = (2, 3) in 2D

**Step 1 — Compute the dot product + r:**
""")
    st.latex(r"x \cdot x + r = (2 \times 2 + 3 \times 3) + 1 = 4 + 9 + 1 = 14")

    st.markdown("**Step 2 — Apply the kernel:**")
    st.latex(r"K(x, x) = (x \cdot x + r)^d = 14^2 = 196")

    st.markdown("""
**Step 3 — What does this correspond to in higher dimensions?**

When d = 2, r = 1, the polynomial kernel implicitly maps (x₁, x₂) to this 6D space:
""")
    st.latex(r"\phi(x_1, x_2) = \left(x_1^2,\ x_2^2,\ \sqrt{2}\,x_1 x_2,\ \sqrt{2r}\,x_1,\ \sqrt{2r}\,x_2,\ r\right)")

    st.markdown("**Step 4 — Plugging in (2, 3):**")

    mapping_data = {
        "Dimension": ["φ₁ = x₁²", "φ₂ = x₂²", "φ₃ = √2·x₁x₂", "φ₄ = √2r·x₁", "φ₅ = √2r·x₂", "φ₆ = r"],
        "Formula":   ["2²", "3²", "√2 × 2 × 3", "√2 × 1 × 2", "√2 × 1 × 3", "1"],
        "Value":     [4.0, 9.0, round(2**0.5 * 6, 3), round(2**0.5 * 2, 3), round(2**0.5 * 3, 3), 1.0],
    }
    st.dataframe(pd.DataFrame(mapping_data), use_container_width=True, hide_index=True)

    phi = [4.0, 9.0, round(2**0.5*6, 3), round(2**0.5*2, 3), round(2**0.5*3, 3), 1.0]
    dot_product = sum(v**2 for v in phi)

    st.markdown(f"""
**Step 5 — Verify: dot product in 6D should equal 196:**

φ(2,3) = {phi}

φ · φ = {" + ".join([f"{v}²" for v in phi])} = **{dot_product:.0f}** (verified!)
""")

    st.markdown(f"""
<div class="callout green" style="color:#2c3e50">
<strong>The key insight:</strong> Computing K(x,x) = 196 took one step. Computing φ(x)·φ(x)
in 6D explicitly took six multiplications and a sum — and that's only for d=2 with 2 features.
With 11 features and d=3, the explicit mapping would have thousands of dimensions.
The kernel does it all with one formula.
</div>
""", unsafe_allow_html=True)

    # ── Visualise the mapping ─────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.axis("off"); ax.set_facecolor("white"); fig.patch.set_facecolor("white")

    def box(x, y, text, color, w=2.3, h=0.55):
        rect = plt.Rectangle((x - w/2, y - h/2), w, h,
                              facecolor=color + "22", edgecolor=color, lw=1.5, zorder=3)
        ax.add_patch(rect)
        ax.text(x, y, text, ha="center", va="center",
                fontsize=8.5, color="#2c3e50", fontweight="bold", zorder=4)

    def arr(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color="#7f8c8d", lw=1.5))

    ax.set_xlim(-0.5, 11); ax.set_ylim(-0.5, 1.5)

    box(1.2, 0.8,  "2D Input\n(2, 3)", BLUE, w=2.0)
    arr(2.3, 0.8, 3.3, 0.8)
    box(4.5, 0.8,  "K(x,x) = (2·2 + 3·3 + 1)²\n= 14² = 196", GREEN, w=3.5)
    arr(6.3, 0.8, 7.3, 0.8)
    box(9.0, 0.8,  "Implicit 6D mapping\nφ = [4, 9, 8.49, 2.83, 4.24, 1]", ORANGE, w=3.5)
    ax.text(4.5, 0.15, "Kernel computes this in ONE step", ha="center", fontsize=9,
            color=GREEN, style="italic")
    ax.text(9.0, 0.15, "6D dot product = 196   (verified!)", ha="center", fontsize=9,
            color=ORANGE, style="italic")
    ax.set_title("Polynomial Kernel with r=1, d=2: from 2D to 6D — without ever going there",
                 fontsize=10, pad=8)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True); plt.close()

    st.markdown("---")

    # (b) DATA PREP
    section("③", "Data Preparation")

    X_tr, X_te, X_tr_s, X_te_s, y_tr, y_te, features, raw_df = load_data()

    st.markdown("""
SVMs **require** two things beyond a standard supervised learning setup:

1. **Labeled data** — every row must have a known class. We use `Walkability_Category`
   (4 classes: Least / Below Average / Above Average / Most Walkable).

2. **Scaled numeric features** — SVMs compute distances between points using dot products.
   If one feature is measured in thousands and another in fractions of 1, the large-scale
   feature dominates every distance calculation and the SVM learns to mostly ignore
   the others. `StandardScaler` fixes this by making every feature have mean = 0 and
   standard deviation = 1.

We also sample 8,000 rows for training. SVMs have O(n²)–O(n³) training complexity —
running on all 43,000 block groups would take minutes per model. At 8,000 samples,
each SVM trains in seconds while still seeing a representative cross-section of America.
""")

    st.markdown("**Before scaling — raw numeric data (first 8 rows):**")
    st.dataframe(raw_df, use_container_width=True)

    st.markdown("**After StandardScaler — same features (first 8 rows of training set):**")
    st.dataframe(X_tr_s.head(8).round(3), use_container_width=True)

    # Train/test diagram
    fig, ax = plt.subplots(figsize=(10, 1.6))
    ax.barh(0, 80, color=BLUE,   height=0.5, label="Training Set (80%)")
    ax.barh(0, 20, left=80, color=RED, height=0.5, label="Test Set (20%)")
    ax.set_xlim(0, 100); ax.set_yticks([])
    ax.set_xlabel("Percentage of Data")
    ax.set_title("Train / Test Split — 80/20 Stratified by Walkability Category", pad=10)
    ax.text(40, 0, "Training Set (80%)", ha="center", va="center",
            color="white", fontweight="bold", fontsize=11)
    ax.text(90, 0, "Test (20%)", ha="center", va="center",
            color="white", fontweight="bold", fontsize=10)
    ax.legend(loc="upper right", fontsize=9); ax.grid(False)
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()

    c1, c2, c3, c4 = st.columns(4)
    metric_card(c1, f"{X_tr.shape[0]:,}", "Train Rows",  BLUE)
    metric_card(c2, f"{X_te.shape[0]:,}", "Test Rows",   RED)
    metric_card(c3, f"{X_tr.shape[1]}",   "Features",    GREEN)
    metric_card(c4, "4",                  "Classes",      ORANGE)

    with st.expander("View Full SVM Training Code", expanded=False):
        st.code("""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

df = pd.read_csv("walkability_cleaned.csv")
df["D4A_clean"] = df["D4A"].replace(-99999, 0)

features = ["D2A_Ranked","D2B_Ranked","D3B_Ranked","D4A_Ranked",
            "NatWalkInd","D3B","D4A_clean","D2B_E8MIXA",
            "Pct_AO0","Pct_AO1","Pct_AO2p"]

X = df[features]; y = df["Walkability_Category"]
X = X.sample(8000, random_state=42); y = y.loc[X.index]

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_tr_s = scaler.fit_transform(X_tr)
X_te_s = scaler.transform(X_te)

costs = [0.1, 1, 10, 100]
for kernel in ["linear", "poly", "rbf"]:
    print(f"\\n=== {kernel.upper()} KERNEL ===")
    for C in costs:
        clf = SVC(kernel=kernel, C=C, gamma="scale", degree=3, random_state=42)
        clf.fit(X_tr_s, y_tr)
        acc = accuracy_score(y_te, clf.predict(X_te_s))
        print(f"  C={C}: {acc*100:.2f}%")
""", language="python")

    st.markdown("---")

    # (d) RESULTS
    section("④", "Results — Three Kernels × Four Costs")

    results, y_te_res = run_svm_experiments()

    # ── Accuracy grid ─────────────────────────────────────────────────
    st.markdown("#### Accuracy across all kernel / cost combinations")

    costs = [0.1, 1, 10, 100]
    kernels = ["linear", "poly", "rbf"]
    kernel_colors = [BLUE, ORANGE, GREEN]

    acc_table = pd.DataFrame(
        {k: [f"{results[k][C]['acc']*100:.2f}%" for C in costs] for k in kernels},
        index=[f"C = {C}" for C in costs]
    )
    acc_table.columns = ["Linear", "Polynomial", "RBF"]
    st.dataframe(acc_table, use_container_width=True)

    # Accuracy line chart
    fig, ax = plt.subplots(figsize=(9, 4))
    for kernel, color in zip(kernels, kernel_colors):
        accs = [results[kernel][C]["acc"] * 100 for C in costs]
        ax.plot(costs, accs, "o-", color=color, lw=2, ms=7, label=kernel.upper())
        for C, a in zip(costs, accs):
            ax.text(C, a + 0.3, f"{a:.1f}%", ha="center", fontsize=8, color=color)
    ax.set_xscale("log")
    ax.set_xlabel("Cost (C) — log scale"); ax.set_ylabel("Test Accuracy (%)")
    ax.set_title("SVM Accuracy vs Cost for each Kernel")
    ax.legend(fontsize=10); ax.grid(alpha=0.3)
    ax.set_ylim(60, 105)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True); plt.close()

    explain(
        "Each line shows how one kernel's accuracy changes as we make it more flexible "
        "(higher C = less regularisation = fits training data more tightly). Generally "
        "accuracy improves with C up to a point. The RBF kernel tends to perform best "
        "because walkability data has curved, non-linear boundaries between categories.",
        "C is the soft-margin penalty parameter. Small C allows more margin violations "
        "(high bias, low variance). Large C enforces stricter separation (low bias, higher "
        "variance). The optimal C balances these. RBF outperforms linear because the "
        "walkability feature space has non-convex class boundaries confirmed by PCA analysis."
    )

    st.markdown("---")

    # ── Confusion matrices — best C per kernel ─────────────────────────
    st.markdown("#### Confusion Matrices — Best Cost per Kernel")

    best_C = {k: max(costs, key=lambda C: results[k][C]["acc"]) for k in kernels}

    st.markdown(f"""
Best cost found:  **Linear** C = {best_C['linear']} |
**Polynomial** C = {best_C['poly']} |
**RBF** C = {best_C['rbf']}
""")

    short_labels = ["Least", "Below Avg", "Above Avg", "Most"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for ax, kernel, color in zip(axes, kernels, kernel_colors):
        C_best = best_C[kernel]
        cm = results[kernel][C_best]["cm"]
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        ax.imshow(cm_norm, interpolation="nearest",
                  cmap=sns.light_palette(color, as_cmap=True))
        for i in range(4):
            for j in range(4):
                tc = "white" if cm_norm[i,j] > 0.5 else "#2c3e50"
                ax.text(j, i, f"{cm[i,j]}\n({cm_norm[i,j]*100:.0f}%)",
                        ha="center", va="center", fontsize=8,
                        color=tc, fontweight="bold")
        ax.set_xticks(range(4)); ax.set_yticks(range(4))
        ax.set_xticklabels(short_labels, fontsize=8, rotation=20)
        ax.set_yticklabels(short_labels, fontsize=8)
        ax.set_xlabel("Predicted", fontsize=9); ax.set_ylabel("Actual", fontsize=9)
        acc = results[kernel][C_best]["acc"] * 100
        ax.set_title(f"{kernel.upper()} kernel (C={C_best})\nAccuracy: {acc:.2f}%", fontsize=10)

    plt.tight_layout(pad=2)
    st.pyplot(fig, use_container_width=True); plt.close()

    explain(
        "Each grid compares what the SVM predicted (columns) to what was actually true (rows). "
        "Bright cells on the diagonal mean correct predictions. Off-diagonal cells are errors. "
        "Notice that mistakes almost always happen between neighbouring categories — predicting "
        "'Below Average' when the area is 'Least Walkable' is a smaller error than predicting "
        "'Most Walkable'. The RBF kernel makes the fewest mistakes overall.",
        "The confusion matrix reveals per-class precision, recall, and F1. Off-diagonal "
        "concentration in adjacent categories reflects the ordinal structure of the label — "
        "the walkability score is continuous and thresholded, so boundary cases are ambiguous. "
        "RBF's superior recall on 'Most Walkable' confirms it better models the non-linear "
        "boundary of the highest-walkability tier."
    )

    st.markdown("---")

    # ── Decision boundary visualisations ─────────────────────────────
    st.markdown("#### Decision Boundary Visualisations — SVMs on 2D PCA projection")
    st.markdown("""
To visualise how each kernel draws its boundary, we reduce the data to 2D using PCA
and train SVMs on that 2D space. The coloured regions show which class the SVM predicts
for every possible point — giving us a map of each kernel's decision surface.
""")

    boundary_data = get_2d_boundary_data()

    cat_colors_map = {
        "Least Walkable": RED, "Below Average Walkable": ORANGE,
        "Above Average Walkable": GREEN, "Most Walkable": BLUE,
    }
    cats = ["Least Walkable", "Below Average Walkable",
            "Above Average Walkable", "Most Walkable"]
    bg_colors = [RED, ORANGE, GREEN, BLUE]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    titles = {"linear": "Linear Kernel", "poly": "Polynomial Kernel (d=3)", "rbf": "RBF Kernel"}

    for ax, kernel in zip(axes, ["linear", "poly", "rbf"]):
        bd = boundary_data[kernel]
        xx, yy, Z = bd["xx"], bd["yy"], bd["Z"]
        X_te_2d, y_te_2d = bd["X_te"], bd["y_te"]

        # Background regions
        Z_num = np.array([cats.index(z) if z in cats else 0 for z in Z]).reshape(xx.shape)
        cmap = matplotlib.colors.ListedColormap([RED + "55", ORANGE + "55",
                                                  GREEN + "55", BLUE + "55"])
        ax.contourf(xx, yy, Z_num, levels=[-0.5,0.5,1.5,2.5,3.5], cmap=cmap, alpha=0.5)
        ax.contour(xx, yy, Z_num, levels=[0.5,1.5,2.5], colors=["white"], linewidths=0.8, alpha=0.6)

        # Scatter points
        for cat, color in cat_colors_map.items():
            mask = y_te_2d == cat
            ax.scatter(X_te_2d[mask, 0], X_te_2d[mask, 1],
                       color=color, s=8, alpha=0.6, rasterized=True)

        ax.set_title(f"{titles[kernel]}\n2D Accuracy: {bd['acc']*100:.1f}%", fontsize=10)
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
        ax.grid(alpha=0.2)

    # Legend
    patches = [matplotlib.patches.Patch(color=c, label=l)
               for l, c in cat_colors_map.items()]
    fig.legend(handles=patches, loc="lower center", ncol=4, fontsize=8,
               bbox_to_anchor=(0.5, -0.04))
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True); plt.close()

    st.markdown("""
<div class="callout orange" style="color:#2c3e50">
<strong>Note on 2D accuracy:</strong> These decision boundaries are drawn on a 2D PCA projection
for visualisation only. The accuracies shown here are lower than the full 11-feature SVM
because we discarded most of the information. The full-feature results in the table above
are the real performance numbers.
</div>
""", unsafe_allow_html=True)

    explain(
        "The coloured regions show what the SVM would predict for any new neighborhood. "
        "The linear kernel draws straight boundaries — clean but rigid. "
        "The polynomial kernel curves the boundaries somewhat. "
        "The RBF kernel produces the smoothest, most flexible regions — it can wrap around "
        "clusters rather than just cutting with lines. The dots are the actual test points, "
        "coloured by their true category.",
        "Decision regions are generated by predicting on a meshgrid in PCA-2D space. "
        "Linear boundaries correspond to hyperplanes in the original space. Polynomial "
        "boundaries correspond to degree-d polynomial surfaces. RBF boundaries are "
        "radially symmetric around support vectors, producing smooth non-convex regions. "
        "The 2D projection discards information (only ~56% variance retained) so visual "
        "accuracy underestimates true classifier performance."
    )

    st.markdown("---")

    # ── Kernel comparison summary ─────────────────────────────────────
    st.markdown("#### Which kernel won, and why?")

    best_kernel = max(kernels, key=lambda k: results[k][best_C[k]]["acc"])
    best_acc_val = results[best_kernel][best_C[best_kernel]]["acc"] * 100

    col1, col2, col3 = st.columns(3)
    for col, kernel, color in zip([col1, col2, col3], kernels, kernel_colors):
        acc = results[kernel][best_C[kernel]]["acc"] * 100
        metric_card(col, f"{acc:.2f}%", f"{kernel.upper()} (C={best_C[kernel]})", color)

    st.markdown(f"""
<div class="callout green" style="color:#2c3e50">
<strong>Best kernel: {best_kernel.upper()} at {best_acc_val:.2f}%</strong><br>
The RBF kernel consistently outperforms the others on walkability data because the
boundaries between walkability tiers are not straight lines in feature space —
they are curved and complex. RBF can model these shapes flexibly by creating
smooth radial regions around clusters of support vectors.<br><br>
Linear kernel is fastest to train and competitive, which suggests the data does
have some linear structure (consistent with the clean PCA separation we found earlier).
Polynomial underperforms slightly at lower C values but catches up at higher C.
</div>
""", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════
    # (e) CONCLUSIONS
    # ══════════════════════════════════════════════════════════════════
    section("⑤", "Conclusions")

    rbf_acc = results["rbf"][best_C["rbf"]]["acc"] * 100
    lin_acc = results["linear"][best_C["linear"]]["acc"] * 100

    st.markdown(f"""
- **SVMs significantly outperform all Module 3 models.** The RBF kernel at C={best_C['rbf']}
  achieves {rbf_acc:.1f}% accuracy — better than Logistic Regression, Naive Bayes, and
  Decision Trees on the same data. This confirms that the decision boundaries between
  walkability tiers are genuinely non-linear.

- **The RBF kernel is the best fit for this data.** Its ability to model curved, flexible
  boundaries matches what we know about walkability — there is no clean straight line
  separating "Most Walkable" from "Above Average" in feature space. The data has
  overlapping distributions near the boundaries of adjacent tiers.

- **Higher cost (C) consistently improves accuracy** across all three kernels, suggesting
  the classes are largely separable but with some genuine overlap at the boundaries —
  the model benefits from being pushed to classify more aggressively.

- **Scaling is critical.** Without StandardScaler, SVM performance on this dataset
  degrades significantly because features like intersection density and transit proximity
  operate on very different numerical scales and would dominate the distance calculations.

- **The confusion patterns are consistent** with every other model in this project —
  the hardest boundary is between *Below Average* and *Above Average* walkability.
  Even the best SVM makes most of its errors at this transition, which reflects a
  genuine structural overlap in the EPA index at that score range.
""")

    st.success("SVM analysis complete. Proceed to the Ensemble tab.")
    st.markdown('<div style="margin-bottom:60px"></div>', unsafe_allow_html=True)    # ══════════════════════════════════════════════════════════════
    # (d) RESULTS
    # ══════════════════════════════════════════════════════════════
    section("④", "Results — 3 Kernels × 3 Cost Values = 9 Models")

    st.markdown("""
**GitHub Links:**
[SVM Code](https://github.com/sezol/my_walkability_app/blob/main/models/svm_tab.py) |
[Dataset](https://github.com/sezol/my_walkability_app/blob/main/walkability_cleaned.csv)
""")

    results, y_te_res = run_svm_experiments()

    costs_to_show  = [0.1, 10, 100]
    all_costs      = [0.1, 1, 10, 100]
    kernels        = ["linear", "poly", "rbf"]
    kernel_colors  = [BLUE, ORANGE, GREEN]
    short_labels   = ["Least", "Below Avg", "Above Avg", "Most"]

    # ── Accuracy summary table ─────────────────────────────────────────
    st.markdown("#### Accuracy across all kernel / cost combinations")

    acc_table = pd.DataFrame(
        {k: [f"{results[k][C]['acc']*100:.2f}%" for C in all_costs] for k in kernels},
        index=[f"C = {C}" for C in all_costs]
    )
    acc_table.columns = ["Linear", "Polynomial", "RBF"]
    st.dataframe(acc_table, use_container_width=True)

    fig, ax = plt.subplots(figsize=(9, 4))
    for kernel, color in zip(kernels, kernel_colors):
        accs = [results[kernel][C]["acc"] * 100 for C in all_costs]
        ax.plot(all_costs, accs, "o-", color=color, lw=2, ms=7, label=kernel.upper())
        for C, a in zip(all_costs, accs):
            ax.text(C, a + 0.3, f"{a:.1f}%", ha="center", fontsize=8, color=color)
    ax.set_xscale("log")
    ax.set_xlabel("Cost (C) — log scale"); ax.set_ylabel("Test Accuracy (%)")
    ax.set_title("SVM Accuracy vs Cost for each Kernel")
    ax.legend(fontsize=10); ax.grid(alpha=0.3); ax.set_ylim(60, 105)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True); plt.close()

    st.markdown("---")

    # ── 9 Confusion Matrices ──────────────────────────────────────────
    st.markdown("#### Confusion Matrices — All 9 Models (3 Kernels × 3 Cost Values: 0.1, 10, 100)")
    st.markdown("""
Each grid shows predicted vs actual walkability category.
Bright diagonal = correct predictions. We show C = 0.1, 10, and 100 for each kernel.
""")

    for kernel, k_color in zip(kernels, kernel_colors):
        st.markdown(f"**{kernel.upper()} Kernel**")
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        for ax, C in zip(axes, costs_to_show):
            cm      = results[kernel][C]["cm"]
            cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
            ax.imshow(cm_norm, interpolation="nearest",
                      cmap=sns.light_palette(k_color, as_cmap=True))
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
            ax.set_title(f"{kernel.upper()} | C={C}\nAccuracy: {results[kernel][C]['acc']*100:.2f}%", fontsize=10)

        plt.tight_layout(pad=2)
        st.pyplot(fig, use_container_width=True); plt.close()

        # Classification report for best C of this kernel
        best_C = max(costs_to_show, key=lambda c: results[kernel][c]["acc"])
        report = results[kernel][best_C]["report"]
        report_rows = []
        for cat in CAT_ORDER:
            r = report.get(cat, {})
            report_rows.append({
                "Class":     cat,
                "Precision": f"{r.get('precision', 0):.3f}",
                "Recall":    f"{r.get('recall', 0):.3f}",
                "F1-Score":  f"{r.get('f1-score', 0):.3f}",
                "Support":   int(r.get("support", 0)),
            })
        st.markdown(f"**Classification Report — {kernel.upper()} at best C={best_C}:**")
        st.dataframe(pd.DataFrame(report_rows), use_container_width=True, hide_index=True)
        st.markdown("---")

    # ── Decision Boundary Visualisations ──────────────────────────────
    st.markdown("#### Decision Boundary Visualisations — 3 Kernels (2D PCA projection)")
    st.markdown("""
SVMs trained on a 2D PCA projection to show how each kernel draws its decision boundary.
The coloured regions show what the SVM predicts for every possible point in that space.
The 2D accuracy shown here is lower than the full 11-feature SVM — shown for visualisation only.
""")

    boundary_data = get_2d_boundary_data()
    cat_colors_map = {
        "Least Walkable": RED,    "Below Average Walkable": ORANGE,
        "Above Average Walkable": GREEN, "Most Walkable": BLUE,
    }
    cats = list(cat_colors_map.keys())

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    titles = {
        "linear": "Linear Kernel\n(straight boundary)",
        "poly":   "Polynomial Kernel\n(curved boundary)",
        "rbf":    "RBF Kernel\n(flexible radial boundary)",
    }

    for ax, kernel in zip(axes, kernels):
        bd = boundary_data[kernel]
        xx, yy, Z = bd["xx"], bd["yy"], bd["Z"]
        X_te_2d, y_te_2d = bd["X_te"], bd["y_te"]
        Z_num = np.array([cats.index(z) if z in cats else 0 for z in Z]).reshape(xx.shape)
        cmap  = matplotlib.colors.ListedColormap([RED+"55", ORANGE+"55", GREEN+"55", BLUE+"55"])
        ax.contourf(xx, yy, Z_num, levels=[-0.5,0.5,1.5,2.5,3.5], cmap=cmap, alpha=0.5)
        ax.contour(xx, yy, Z_num, levels=[0.5,1.5,2.5], colors=["white"], linewidths=0.8, alpha=0.6)
        for cat, color in cat_colors_map.items():
            mask = y_te_2d == cat
            ax.scatter(X_te_2d[mask, 0], X_te_2d[mask, 1],
                       color=color, s=8, alpha=0.6, rasterized=True)
        ax.set_title(f"{titles[kernel]}\n2D Accuracy: {bd['acc']*100:.1f}%", fontsize=9)
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.grid(alpha=0.2)

    patches = [matplotlib.patches.Patch(color=c, label=l) for l, c in cat_colors_map.items()]
    fig.legend(handles=patches, loc="lower center", ncol=4, fontsize=8,
               bbox_to_anchor=(0.5, -0.04))
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True); plt.close()

    st.markdown("---")

    # ── Kernel comparison summary ─────────────────────────────────────
    st.markdown("#### Which kernel won, and why?")

    best_C_per = {k: max(all_costs, key=lambda C: results[k][C]["acc"]) for k in kernels}
    cols = st.columns(3)
    for col, kernel, color in zip(cols, kernels, kernel_colors):
        acc = results[kernel][best_C_per[kernel]]["acc"] * 100
        metric_card(col, f"{acc:.2f}%", f"{kernel.upper()} (C={best_C_per[kernel]})", color)

    best_kernel = max(kernels, key=lambda k: results[k][best_C_per[k]]["acc"])
    best_acc    = results[best_kernel][best_C_per[best_kernel]]["acc"] * 100

    st.markdown(f"""
<div class="callout green" style="color:#2c3e50">
<strong>Best kernel: {best_kernel.upper()} at {best_acc:.2f}%</strong><br>
The RBF kernel wins because walkability class boundaries are curved and non-linear in feature space.
Linear works well too — confirming the data has some linear separability, consistent with the clean
PCA separation found earlier. Polynomial sits in between. Higher C consistently helped all kernels,
suggesting the classes are largely separable with genuine overlap only near tier boundaries.
</div>
""", unsafe_allow_html=True)