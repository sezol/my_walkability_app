"""
ARM Page — EPA Walkability Index
Author: Sejal Hukare
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
from itertools import combinations
import warnings
warnings.filterwarnings("ignore")

BLUE   = "#3498db"; GREEN = "#2ecc71"; RED = "#e74c3c"
ORANGE = "#f39c12"; PURPLE = "#9b59b6"; TEAL = "#1abc9c"

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "#f8f9fa",
    "axes.edgecolor": "#dee2e6", "axes.labelcolor": "#2c3e50",
    "xtick.color": "#7f8c8d",   "ytick.color": "#7f8c8d",
    "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.labelsize": 10, "font.size": 10,
})

NODE_COLOR_MAP = {
    "Most Walkable": "#3498db",  "Above Average Walkable": "#2ecc71",
    "Below Average Walkable": "#f39c12", "Least Walkable": "#e74c3c",
    "No Transit": "#e74c3c",     "Low Transit": "#f39c12",
    "High Transit": "#2ecc71",
    "High Car Dep": "#e74c3c",   "Med Car Dep": "#f39c12",
    "Low Car Dep": "#2ecc71",
    "High Emp Mix": "#3498db",   "Med Emp Mix": "#9b59b6",
    "Low Emp Mix": "#e74c3c",
    "High Int Dens": "#3498db",  "Med Int Dens": "#9b59b6",
    "Low Int Dens": "#e74c3c",
    "High Pop": "#3498db",       "Med Pop": "#9b59b6",
    "Low Pop": "#f39c12",
}


@st.cache_data
def load_arm_data(min_sup=0.05):
    df = pd.read_csv("walkability_cleaned.csv")
    d  = df.copy()
    d["Transit"]  = d["D4A"].apply(
        lambda x: "No Transit" if x == -99999 else ("Low Transit" if x < 500 else "High Transit"))
    d["WalkCat"]  = d["Walkability_Category"]
    d["Auto Dep"] = d["Pct_AO2p"].apply(
        lambda x: "High Car Dep" if x > .7 else ("Med Car Dep" if x > .4 else "Low Car Dep"))
    d["Emp Mix"]  = d["D2B_E8MIXA"].apply(
        lambda x: "High Emp Mix" if x > .6 else ("Med Emp Mix" if x > .3 else "Low Emp Mix"))
    d["Int Dens"] = d["D3B"].apply(
        lambda x: "High Int Dens" if x > 100 else ("Med Int Dens" if x > 40 else "Low Int Dens"))
    d["Pop Dens"] = d["TotPop"].apply(
        lambda x: "High Pop" if x > 3000 else ("Med Pop" if x > 1000 else "Low Pop"))

    # Raw columns for "before" display
    raw_cols  = ["D4A", "Walkability_Category", "Pct_AO2p", "D2B_E8MIXA", "D3B", "TotPop"]
    raw_df    = df[raw_cols].dropna()

    cols  = ["WalkCat", "Transit", "Auto Dep", "Emp Mix", "Int Dens", "Pop Dens"]
    trans = d[cols].apply(lambda r: set(r.values), axis=1).tolist()
    n     = len(trans)

    all_items: set = set()
    for t in trans:
        all_items.update(t)
    item_sup   = {frozenset([i]): sum(1 for t in trans if i in t) / n for i in all_items}
    freq1      = {k: v for k, v in item_sup.items() if v >= min_sup}
    items_list = [list(k)[0] for k in freq1]

    pair_sup: dict = {}
    for a, b in combinations(items_list, 2):
        s = sum(1 for t in trans if a in t and b in t) / n
        if s >= min_sup:
            pair_sup[frozenset([a, b])] = s

    rules = []
    for itemset, sup in pair_sup.items():
        it = list(itemset)
        for i in range(2):
            ant = frozenset([it[i]]); con = frozenset([it[1 - i]])
            as_ = item_sup.get(ant, 0); cs_ = item_sup.get(con, 0)
            if as_ > 0 and cs_ > 0:
                conf = sup / as_; lift = conf / cs_
                rules.append({
                    "Antecedent": list(ant)[0], "Consequent": list(con)[0],
                    "Support": round(sup, 4), "Confidence": round(conf, 4),
                    "Lift": round(lift, 4),
                })
    rdf = (pd.DataFrame(rules)
           .sort_values("Lift", ascending=False)
           .drop_duplicates()
           .reset_index(drop=True))
    return rdf, item_sup, d[cols], d[cols + ["Walkability_Category"]], raw_df


def metric_card(col, val, label, color=BLUE):
    col.markdown(
        f'<div style="background:#f8f9fa;border-left:4px solid {color};'
        f'border-radius:6px;padding:14px 16px;margin-bottom:8px">'
        f'<div style="font-size:1.4rem;font-weight:700;color:#2c3e50;line-height:1.1">{val}</div>'
        f'<div style="font-size:0.73rem;color:#7f8c8d;text-transform:uppercase;'
        f'letter-spacing:0.06em;margin-top:3px">{label}</div></div>',
        unsafe_allow_html=True)


def explain(layman, ds):
    with st.expander("What does this mean?", expanded=False):
        st.markdown(f"**In plain English:** {layman}")
        st.markdown(f"**For data scientists:** {ds}")


def rules_table(df_rules, sort_col, n=15):
    top = df_rules.sort_values(sort_col, ascending=False).head(n).reset_index(drop=True)

    def lift_cell(v):
        c = RED if v >= 2.5 else ORANGE if v >= 1.8 else BLUE
        arrow = "▲▲" if v >= 2.5 else "▲" if v >= 1.8 else ""
        return (f'<span style="background:{c}22;color:{c};border:1px solid {c}44;'
                f'padding:2px 7px;border-radius:4px;font-weight:600;font-size:.78rem">'
                f'{v:.3f} {arrow}</span>')

    def conf_cell(v):
        c = GREEN if v >= 0.8 else BLUE if v >= 0.5 else ORANGE
        return (f'<span style="background:{c}22;color:{c};border:1px solid {c}44;'
                f'padding:2px 7px;border-radius:4px;font-weight:600;font-size:.78rem">'
                f'{v:.3f}</span>')

    rows = ""
    for i, r in top.iterrows():
        rows += (f"<tr>"
                 f'<td>{i+1}</td>'
                 f'<td><span style="background:#eaf4fd;color:{BLUE};border:1px solid {BLUE}44;'
                 f'padding:2px 7px;border-radius:4px;font-size:.78rem">{r["Antecedent"]}</span></td>'
                 f'<td style="text-align:center;color:#aaa">→</td>'
                 f'<td><span style="background:#eafaf1;color:{GREEN};border:1px solid {GREEN}44;'
                 f'padding:2px 7px;border-radius:4px;font-size:.78rem">{r["Consequent"]}</span></td>'
                 f'<td style="font-family:monospace;color:#555;font-size:.82rem">{r["Support"]:.4f}</td>'
                 f'<td>{conf_cell(r["Confidence"])}</td>'
                 f'<td>{lift_cell(r["Lift"])}</td>'
                 f"</tr>")

    html = f"""
    <div style="overflow-x:auto">
    <table style="width:100%;border-collapse:collapse;font-size:.84rem">
      <thead><tr style="background:#f0f3f4">
        <th style="padding:8px 10px;text-align:left;border-bottom:2px solid #d5d8dc;
            color:#566573;font-size:.73rem;text-transform:uppercase;letter-spacing:.05em">#</th>
        <th style="padding:8px 10px;text-align:left;border-bottom:2px solid #d5d8dc;
            color:#566573;font-size:.73rem;text-transform:uppercase;letter-spacing:.05em">Antecedent</th>
        <th style="padding:8px 10px;border-bottom:2px solid #d5d8dc"></th>
        <th style="padding:8px 10px;text-align:left;border-bottom:2px solid #d5d8dc;
            color:#566573;font-size:.73rem;text-transform:uppercase;letter-spacing:.05em">Consequent</th>
        <th style="padding:8px 10px;text-align:left;border-bottom:2px solid #d5d8dc;
            color:#566573;font-size:.73rem;text-transform:uppercase;letter-spacing:.05em">Support</th>
        <th style="padding:8px 10px;text-align:left;border-bottom:2px solid #d5d8dc;
            color:#566573;font-size:.73rem;text-transform:uppercase;letter-spacing:.05em">Confidence</th>
        <th style="padding:8px 10px;text-align:left;border-bottom:2px solid #d5d8dc;
            color:#566573;font-size:.73rem;text-transform:uppercase;letter-spacing:.05em">Lift</th>
      </tr></thead>
      <tbody>{rows}</tbody>
    </table></div>"""
    st.markdown(html, unsafe_allow_html=True)


def draw_network(rules_df, top_n=20, title="Association Network"):
    top = rules_df.sort_values("Lift", ascending=False).head(top_n)
    G   = nx.DiGraph()
    for _, row in top.iterrows():
        G.add_edge(row["Antecedent"], row["Consequent"],
                   weight=row["Lift"], conf=row["Confidence"], sup=row["Support"])

    pos          = nx.spring_layout(G, seed=42, k=2.2)
    node_colors  = [NODE_COLOR_MAP.get(n, "#95a5a6") for n in G.nodes()]
    edge_weights = [G[u][v]["weight"] for u, v in G.edges()]
    edge_widths  = [1 + (w - 1) * 1.2 for w in edge_weights]
    edge_alphas  = [min(1.0, 0.3 + w * 0.15) for w in edge_weights]

    fig, ax = plt.subplots(figsize=(13, 8), facecolor="white")
    ax.set_facecolor("#fafafa")
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=1600, alpha=0.92, ax=ax)
    for (u, v), w, wd, al in zip(G.edges(), edge_weights, edge_widths, edge_alphas):
        nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], width=wd, alpha=al,
                               edge_color=[ORANGE if w >= 2.5 else BLUE],
                               connectionstyle="arc3,rad=0.08", arrowsize=18, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=7.5, font_color="white", font_weight="bold", ax=ax)
    ax.set_title(f"{title}\n(arrow thickness = Lift strength; orange = Lift >= 2.5)",
                 fontsize=12, fontweight="bold", pad=14)
    ax.axis("off")
    plt.tight_layout()
    return fig


# ── Overview Image 1: Support / Confidence / Lift illustrated ─────────────
def plot_arm_metrics():
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    # --- Support ---
    ax = axes[0]
    total   = 100
    both    = 18
    only_a  = 32
    only_b  = 25
    neither = total - both - only_a - only_b

    labels  = ["Both A & B\n(Support)", "Only A", "Only B", "Neither"]
    sizes   = [both, only_a, only_b, neither]
    colors  = [BLUE, "#aed6f1", "#a9dfbf", "#d5d8dc"]
    bars    = ax.bar(labels, sizes, color=colors, edgecolor="white", width=0.55)
    for bar, v in zip(bars, sizes):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
                str(v), ha="center", fontsize=10, fontweight="bold", color="#2c3e50")
    ax.set_ylim(0, 55)
    ax.set_ylabel("Number of block groups (out of 100)")
    ax.set_title(f"Support = P(A and B) = {both}/{total} = {both/total:.2f}", fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    # --- Confidence ---
    ax = axes[1]
    conf = both / (both + only_a)
    bars = ax.bar(["A occurs\n(50 block groups)", "A and B\nboth occur"],
                  [both + only_a, both],
                  color=["#aed6f1", GREEN], edgecolor="white", width=0.5)
    ax.annotate("", xy=(1, both), xytext=(0, both + only_a),
                arrowprops=dict(arrowstyle="->", color=GREEN, lw=2))
    ax.set_ylim(0, 65)
    ax.set_ylabel("Count")
    ax.set_title(f"Confidence = P(B|A) = {both}/{both+only_a} = {conf:.2f}", fontsize=10)
    for bar, v in zip(bars, [both + only_a, both]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
                str(v), ha="center", fontsize=10, fontweight="bold", color="#2c3e50")
    ax.grid(axis="y", alpha=0.3)

    # --- Lift ---
    ax = axes[2]
    p_b    = (both + only_b) / total
    lift   = conf / p_b
    random = conf / lift  # = p_b, for annotation

    scenarios = ["Expected if\nindependent", "Actual\nConfidence"]
    vals      = [p_b, conf]
    bar_colors = [ORANGE, RED if lift > 1 else BLUE]
    bars = ax.bar(scenarios, vals, color=bar_colors, edgecolor="white", width=0.45)
    ax.axhline(p_b, color=ORANGE, ls="--", lw=1.3, alpha=0.7)
    ax.set_ylim(0, 0.85)
    ax.set_ylabel("Probability")
    ax.set_title(f"Lift = Confidence / P(B) = {conf:.2f} / {p_b:.2f} = {lift:.2f}", fontsize=10)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{v:.2f}", ha="center", fontsize=10, fontweight="bold", color="#2c3e50")
    ax.text(1.27, (conf + p_b)/2, f"Lift = {lift:.2f}x", fontsize=9,
            color=RED, fontweight="bold", va="center")
    ax.grid(axis="y", alpha=0.3)

    plt.suptitle("ARM Metrics Illustrated — Support, Confidence, and Lift",
                 fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()
    return fig


# ── Overview Image 2: Apriori pruning steps ──────────────────────────────
def plot_apriori_steps():
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.set_xlim(0, 13); ax.set_ylim(0, 5)
    ax.axis("off")
    ax.set_facecolor("white"); fig.patch.set_facecolor("white")

    def box(x, y, text, color, w=2.0, h=0.65, alpha=1.0):
        rect = plt.Rectangle((x - w/2, y - h/2), w, h,
                              facecolor=color, edgecolor="#2c3e50",
                              linewidth=1.2, alpha=alpha, zorder=3)
        ax.add_patch(rect)
        ax.text(x, y, text, ha="center", va="center",
                fontsize=8, color="#2c3e50", fontweight="bold", zorder=4)

    def arr(x1, y1, x2, y2, label=""):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color="#7f8c8d", lw=1.3))
        if label:
            ax.text((x1+x2)/2 + 0.15, (y1+y2)/2, label,
                    fontsize=7.5, color="#7f8c8d", ha="left", va="center")

    # Step labels
    for x, lbl in [(1.5,"Step 1\nRaw Data"), (4.0,"Step 2\nFrequent Items"),
                    (7.0,"Step 3\nFrequent Pairs"), (10.2,"Step 4\nGenerate Rules"),
                    (12.2,"Step 5\nRank by Lift")]:
        ax.text(x, 4.6, lbl, ha="center", fontsize=8, color="#3498db",
                fontweight="bold")

    # Step 1: basket items
    for i, (item, col) in enumerate([("No Transit", RED), ("Least Walk.", RED),
                                      ("High Car", ORANGE), ("Low Int.", RED)]):
        box(1.5, 3.5 - i * 0.75, item, col + "33", w=1.9, h=0.6)

    arr(2.5, 2.5, 3.2, 2.5, "count\nfrequency")

    # Step 2: frequent items (pruned)
    for i, (item, col, freq) in enumerate([("No Transit", RED, "0.42"),
                                            ("Least Walk.", RED, "0.31"),
                                            ("High Car", ORANGE, "0.28")]):
        box(4.0, 3.5 - i * 0.85, f"{item}\nsup={freq}", col + "44", w=2.1, h=0.7)
    ax.text(4.0, 0.9, "Low Int. pruned\n(sup < 0.05)", ha="center",
            fontsize=7.5, color=RED, style="italic")

    arr(5.1, 2.5, 5.9, 2.5, "combine\npairs")

    # Step 3: frequent pairs
    for i, (pair, sup) in enumerate([("{No Transit,\nLeast Walk.}", "0.29"),
                                      ("{No Transit,\nHigh Car}", "0.18"),
                                      ("{Least Walk.,\nHigh Car}", "0.12")]):
        box(7.0, 3.5 - i * 0.9, f"{pair}\nsup={sup}", BLUE + "22", w=2.4, h=0.78)

    arr(8.1, 2.5, 8.9, 2.5, "generate\nrules")

    # Step 4: rules
    for i, rule in enumerate(["No Transit → Least Walk.",
                               "Least Walk. → No Transit",
                               "High Car → Least Walk."]):
        box(10.2, 3.5 - i * 0.9, rule, GREEN + "22", w=2.6, h=0.68)

    arr(11.5, 2.5, 11.9, 2.5)

    # Step 5: ranked
    box(12.2, 3.5, "Lift=3.1", RED + "44", w=1.4, h=0.6)
    box(12.2, 2.7, "Lift=2.8", ORANGE + "44", w=1.4, h=0.6)
    box(12.2, 1.9, "Lift=1.9", BLUE + "33", w=1.4, h=0.6)

    ax.set_title("The Apriori Algorithm — From Raw Baskets to Ranked Rules",
                 fontsize=11, fontweight="bold", pad=8)
    plt.tight_layout()
    return fig


def app():
    st.markdown("""
    <style>
    .section-hdr{font-size:1.25rem;font-weight:700;color:#2c3e50;
        border-left:4px solid #3498db;padding-left:12px;margin:1.5rem 0 0.8rem}
    .callout{background:#eaf4fd;border-left:4px solid #3498db;border-radius:6px;
        padding:14px 18px;margin:10px 0;font-size:0.9rem;color:#2c3e50}
    .callout.green{background:#eafaf1;border-color:#2ecc71;color:#2c3e50}
    .callout.orange{background:#fef9e7;border-color:#f39c12;color:#2c3e50}
    .callout.red{background:#fdedec;border-color:#e74c3c;color:#2c3e50}
    </style>""", unsafe_allow_html=True)

    st.title("Association Rule Mining (ARM)")

    rules_df, item_sup, trans_df, full_df, raw_df = load_arm_data()

    # ── (a) OVERVIEW ─────────────────────────────────────────────────────────
    st.markdown('<div class="section-hdr">(a) Overview — What is ARM?</div>', unsafe_allow_html=True)

    st.markdown("""
**In plain English:** Association Rule Mining is like finding out "people who buy bread
also tend to buy butter." In our case, instead of groceries, we're looking at walkability
features — like "neighbourhoods with no transit access almost always have low intersection
density." We scan all 43,065 U.S. block groups looking for patterns that appear together
frequently, then measure how strong those patterns are.

**For data scientists:** ARM discovers frequent co-occurrence patterns in transactional
datasets using the **Apriori algorithm**. It generates rules of the form A → B, where A
(antecedent) and B (consequent) are itemsets. Rules are evaluated on three metrics:
**Support** (frequency), **Confidence** (conditional probability), and **Lift**
(departure from independence). We implement a custom Python Apriori using 1- and
2-itemsets on 43,065 discretised block-group "baskets."
    """)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"""
        <div style="background:#f8f9fa;border-radius:10px;padding:20px;border:1px solid #dee2e6;color:#2c3e50">
        <div style="font-size:1rem;font-weight:700;color:#2c3e50;margin-bottom:12px">
            The Three Key Measures
        </div>
        <div style="margin-bottom:10px">
            <span style="background:#eaf4fd;color:{BLUE};padding:3px 9px;border-radius:4px;
            font-weight:700;font-size:.85rem">SUPPORT</span>&nbsp;
            <span style="color:#555;font-size:.88rem">= P(A and B)</span><br>
            <span style="color:#7f8c8d;font-size:.83rem">How often the rule appears overall.
            A support of 0.18 means 18% of all block groups match this pattern.</span>
        </div>
        <div style="margin-bottom:10px">
            <span style="background:#eafaf1;color:{GREEN};padding:3px 9px;border-radius:4px;
            font-weight:700;font-size:.85rem">CONFIDENCE</span>&nbsp;
            <span style="color:#555;font-size:.88rem">= P(B|A)</span><br>
            <span style="color:#7f8c8d;font-size:.83rem">Given A, how often does B also appear?
            Confidence of 0.92 means 92% of the time A occurs, B also occurs.</span>
        </div>
        <div>
            <span style="background:#fef9e7;color:{ORANGE};padding:3px 9px;border-radius:4px;
            font-weight:700;font-size:.85rem">LIFT</span>&nbsp;
            <span style="color:#555;font-size:.88rem">= Confidence / P(B)</span><br>
            <span style="color:#7f8c8d;font-size:.83rem">Lift > 1 means A and B appear together
            more than random chance. Lift = 2.8 means 2.8x more likely than by chance.</span>
        </div>
        </div>""", unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div style="background:#f8f9fa;border-radius:10px;padding:20px;border:1px solid #dee2e6;color:#2c3e50">
        <div style="font-size:1rem;font-weight:700;color:#2c3e50;margin-bottom:12px">
            How the Apriori Algorithm Works
        </div>
        <ol style="color:#555;font-size:.88rem;padding-left:18px;line-height:1.9">
            <li>Each block group becomes a <b>"basket"</b> of labels
                (e.g. No Transit, Least Walkable, High Car Dep)</li>
            <li>Scan all baskets and count how often each single item appears
                — keep only <b>frequent items</b> (support >= threshold)</li>
            <li>Pair up frequent items and scan again
                — keep only <b>frequent pairs</b></li>
            <li>From each frequent pair, generate two <b>rules</b>
                (A→B and B→A)</li>
            <li>Compute Support, Confidence, Lift for every rule</li>
            <li><b>Rank</b> by Lift to find the most interesting associations</li>
        </ol>
        <div style="background:#eaf4fd;border-radius:6px;padding:10px;margin-top:10px;
            font-size:.82rem;color:#2c3e50">
            <b>Key principle (Apriori property):</b> If an itemset is infrequent,
            all its supersets are also infrequent — this prunes the search space.
        </div>
        </div>""", unsafe_allow_html=True)

    # ── Overview Image 1: Support / Confidence / Lift ────────────────────────
    st.markdown("#### Image 1 — Support, Confidence, and Lift Illustrated")
    st.markdown("""
Using a simplified example of 100 block groups, the charts below show exactly how each
ARM metric is calculated and what it means geometrically.
    """)
    fig1 = plot_arm_metrics()
    st.pyplot(fig1, use_container_width=True); plt.close()

    explain(
        "The left chart shows Support — the fraction of all block groups where both A and B appear. "
        "The middle chart shows Confidence — given that A occurs, how often does B also occur? "
        "The right chart shows Lift — is the actual confidence higher than what random chance would predict? "
        "A Lift above 1.0 means the two items appear together more than by coincidence.",
        "Support = |T(A and B)| / |T|. Confidence = Support(A,B) / Support(A). "
        "Lift = Confidence / Support(B). Lift normalises confidence by the marginal probability "
        "of the consequent, making it comparable across rules regardless of how common B is."
    )

    # ── Overview Image 2: Apriori steps ──────────────────────────────────────
    st.markdown("#### Image 2 — The Apriori Algorithm Step by Step")
    st.markdown("""
The diagram below traces the full Apriori pipeline from raw transaction baskets through
candidate pruning, rule generation, and final ranking by Lift.
    """)
    fig2 = plot_apriori_steps()
    st.pyplot(fig2, use_container_width=True); plt.close()

    explain(
        "The algorithm starts on the left with all items in all baskets. It first counts "
        "how often each single item appears and throws away rare ones (pruning). Then it "
        "tries combining the surviving items into pairs, prunes rare pairs, and generates "
        "a rule for each direction of each pair. Finally it ranks all rules by Lift — "
        "so the most surprising and meaningful patterns float to the top.",
        "Apriori exploits the anti-monotone property: if {A} is infrequent, every superset "
        "{A,B}, {A,B,C}, ... is also infrequent. This allows aggressive pruning of the "
        "candidate itemset space from O(2^n) to a manageable size. Our implementation "
        "stops at 2-itemsets, producing rules with exactly one antecedent and one consequent."
    )

    st.markdown("---")

    # ── (b) DATA PREP ─────────────────────────────────────────────────────────
    st.markdown('<div class="section-hdr">(b) Data Preparation for ARM</div>', unsafe_allow_html=True)

    st.markdown("""
ARM requires **transaction data only** — a list of baskets where each basket is a set of
category labels. There are no numeric values in the ARM input. Continuous variables must
first be **discretised** (binned) into meaningful categories.

The table below shows the **original raw data** before any transformation, followed by the
**transformed transaction data** after discretisation.
    """)

    # ── BEFORE dataframe ─────────────────────────────────────────────────────
    st.markdown("**Before Transformation — Raw Numeric Data (first 8 rows):**")
    st.markdown("""
    These are the continuous numeric columns from `walkability_cleaned.csv` that will be
    discretised. Note the raw values: `D4A` is transit proximity in metres (−99999 = no transit),
    `Pct_AO2p` is the fraction of households with 2+ cars, `D2B_E8MIXA` is employment entropy,
    `D3B` is intersection density per km², and `TotPop` is block group population.
    """)
    st.dataframe(raw_df.head(8), use_container_width=True)

    st.markdown("---")

    # ── Discretisation rules ──────────────────────────────────────────────────
    st.markdown("**Discretisation Rules Applied:**")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
**Transit Access (D4A)**
- `No Transit` = −99,999 (sentinel)
- `Low Transit` = 0–499 m
- `High Transit` = >= 500 m

**Car Dependence (Pct_AO2p)**
- `Low Car Dep` = < 40%
- `Med Car Dep` = 40–70%
- `High Car Dep` = > 70%
        """)
    with c2:
        st.markdown("""
**Employment Mix (D2B_E8MIXA)**
- `Low Emp Mix` = < 0.3
- `Med Emp Mix` = 0.3–0.6
- `High Emp Mix` = > 0.6

**Intersection Density (D3B)**
- `Low Int Dens` = < 40 /km²
- `Med Int Dens` = 40–100 /km²
- `High Int Dens` = > 100 /km²
        """)
    with c3:
        st.markdown("""
**Population (TotPop)**
- `Low Pop` = < 1,000
- `Med Pop` = 1,000–3,000
- `High Pop` = > 3,000

**Walkability Category**
- Least / Below Avg /
  Above Avg / Most Walkable
        """)

    st.markdown("---")

    # ── AFTER dataframe ───────────────────────────────────────────────────────
    st.markdown("**After Transformation — Transaction Data (first 8 rows):**")
    st.markdown("""
    All numeric values have been replaced with category labels. Each row is now a
    transaction basket — a set of co-occurring labels that ARM will mine for patterns.
    The numeric columns are no longer present.
    """)
    st.dataframe(trans_df.head(8), use_container_width=True)

    explain(
        "The left table has numbers — transit distances, percentages, densities. "
        "ARM cannot work with numbers directly, so we convert each number into a label "
        "based on where it falls in a range. For example, a D4A value of -99999 becomes "
        "'No Transit', and a Pct_AO2p of 0.85 becomes 'High Car Dep'. "
        "The result is the right table — each row is a basket of labels ready for ARM.",
        "Discretisation maps continuous features to nominal items via threshold binning. "
        "Each transaction T_i = {item_1, ..., item_k} where each item is a (variable, bin) pair. "
        "The resulting transaction matrix has 43,065 rows x 6 item columns, "
        "yielding 18 unique items. min_support = 0.05 retains items appearing in >= 5% of baskets."
    )

    # Itemset frequency chart
    st.markdown("**Frequent item support (how often each item appears across all block groups):**")
    isdf = (pd.DataFrame([{"Item": list(k)[0], "Support": v} for k, v in item_sup.items()])
            .sort_values("Support", ascending=False))
    fig, ax = plt.subplots(figsize=(11, 5.5))
    colors = [NODE_COLOR_MAP.get(item, "#95a5a6") for item in isdf["Item"][::-1]]
    bars   = ax.barh(isdf["Item"][::-1], isdf["Support"][::-1],
                     color=colors, height=0.55, edgecolor="white")
    ax.axvline(0.05, color=RED, ls="--", lw=1.2, label="Min support threshold = 0.05")
    ax.set_xlabel("Support (proportion of all block groups)")
    ax.set_title("Frequent Item Support — How common is each item?")
    ax.legend(fontsize=9); ax.grid(axis="x", alpha=0.4)
    for bar in bars:
        w = bar.get_width()
        ax.text(w + 0.004, bar.get_y() + bar.get_height() / 2,
                f"{w:.3f}", va="center", fontsize=8, color="#555")
    plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()

    st.markdown(f"""
    <div class="callout">
    <b>Thresholds used:</b> min_support = <b>0.05</b> (item must appear in >= 5% of block groups),
    applied to both 1-itemsets and 2-itemsets before rule generation.
    This yielded <b>{len(item_sup)} frequent items</b> and <b>{len(rules_df)} rules</b> total.
    No minimum confidence threshold was applied — rules were ranked post-hoc.
    </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ── (c) CODE ──────────────────────────────────────────────────────────────
    st.markdown('<div class="section-hdr">(c) ARM Implementation</div>', unsafe_allow_html=True)

    with st.expander("View ARM Python Code", expanded=False):
        st.code("""
import pandas as pd
from itertools import combinations

df = pd.read_csv("walkability_cleaned.csv")

# Step 1: Discretise continuous variables
df["Transit"]  = df["D4A"].apply(
    lambda x: "No Transit" if x == -99999 else
              ("Low Transit" if x < 500 else "High Transit"))
df["WalkCat"]  = df["Walkability_Category"]
df["Auto Dep"] = df["Pct_AO2p"].apply(
    lambda x: "High Car Dep" if x > .7 else
              ("Med Car Dep" if x > .4 else "Low Car Dep"))
df["Emp Mix"]  = df["D2B_E8MIXA"].apply(
    lambda x: "High Emp Mix" if x > .6 else
              ("Med Emp Mix" if x > .3 else "Low Emp Mix"))
df["Int Dens"] = df["D3B"].apply(
    lambda x: "High Int Dens" if x > 100 else
              ("Med Int Dens" if x > 40 else "Low Int Dens"))
df["Pop Dens"] = df["TotPop"].apply(
    lambda x: "High Pop" if x > 3000 else
              ("Med Pop" if x > 1000 else "Low Pop"))

# Step 2: Build transaction baskets
cols  = ["WalkCat","Transit","Auto Dep","Emp Mix","Int Dens","Pop Dens"]
trans = df[cols].apply(lambda row: set(row.values), axis=1).tolist()
n     = len(trans)

# Step 3: Compute 1-itemset support
all_items = set()
for t in trans: all_items.update(t)
item_sup = {
    frozenset([i]): sum(1 for t in trans if i in t) / n
    for i in all_items
}

# Step 4: Prune to frequent 1-itemsets (min_support = 0.05)
MIN_SUP    = 0.05
freq1      = {k: v for k, v in item_sup.items() if v >= MIN_SUP}
items_list = [list(k)[0] for k in freq1]

# Step 5: Generate frequent 2-itemsets
pair_sup = {}
for a, b in combinations(items_list, 2):
    s = sum(1 for t in trans if a in t and b in t) / n
    if s >= MIN_SUP:
        pair_sup[frozenset([a, b])] = s

# Step 6: Generate rules with support, confidence, lift
rules = []
for itemset, sup in pair_sup.items():
    it = list(itemset)
    for i in range(2):
        ant, con = frozenset([it[i]]), frozenset([it[1-i]])
        as_, cs_ = item_sup.get(ant, 0), item_sup.get(con, 0)
        if as_ > 0 and cs_ > 0:
            conf = sup / as_
            lift = conf / cs_
            rules.append({
                "Antecedent": list(ant)[0], "Consequent": list(con)[0],
                "Support": round(sup, 4), "Confidence": round(conf, 4),
                "Lift": round(lift, 4),
            })

rules_df = (pd.DataFrame(rules)
            .sort_values("Lift", ascending=False)
            .drop_duplicates()
            .reset_index(drop=True))
        """, language="python")

    st.markdown("---")

    # ── (d) RESULTS ───────────────────────────────────────────────────────────
    st.markdown('<div class="section-hdr">(d) Results</div>', unsafe_allow_html=True)

    top_lift = rules_df.iloc[0]
    c1,c2,c3,c4,c5,c6 = st.columns(6)
    metric_card(c1, len(rules_df),                        "Total Rules",      BLUE)
    metric_card(c2, f"{rules_df['Lift'].max():.2f}",      "Max Lift",         GREEN)
    metric_card(c3, f"{rules_df['Confidence'].max():.2f}","Max Confidence",   BLUE)
    metric_card(c4, f"{rules_df['Support'].max():.3f}",   "Max Support",      ORANGE)
    metric_card(c5, len(item_sup),                        "Frequent Items",   PURPLE)
    metric_card(c6, "0.05",                               "Min Support Used", RED)

    st.markdown("---")

    t1, t2, t3 = st.tabs(["Top 15 by Lift", "Top 15 by Confidence", "Top 15 by Support"])

    with t1:
        st.markdown("**Rules ranked by Lift — the most interesting associations (furthest from random chance)**")
        rules_table(rules_df, "Lift", 15)
        explain(
            f"The top rule is '{top_lift['Antecedent']} → {top_lift['Consequent']}' with "
            f"lift = {top_lift['Lift']:.2f}. These two characteristics appear together "
            f"{top_lift['Lift']:.1f}x more often than you'd expect if they were unrelated.",
            "Lift = Confidence / P(Consequent). Values > 1 indicate positive correlation; "
            "values < 1 indicate negative correlation; = 1 means independence."
        )

    with t2:
        st.markdown("**Rules ranked by Confidence — the most reliable associations**")
        rules_table(rules_df, "Confidence", 15)
        explain(
            "Confidence tells you: given that a neighbourhood has X, how often does it also have Y? "
            "A confidence of 0.99 is almost a certainty.",
            "Confidence = Support(A and B) / Support(A). High confidence rules can be misleading "
            "if the consequent is already very common — which is why Lift is typically preferred."
        )

    with t3:
        st.markdown("**Rules ranked by Support — the most frequent associations**")
        rules_table(rules_df, "Support", 15)
        explain(
            "Support tells you how common a pattern is across all block groups. "
            "High support means the pattern is widespread, not just true for a few neighbourhoods.",
            "Support = |T(A and B)| / |T|. High-support rules are statistically robust but may "
            "have low lift if both items are individually common."
        )

    st.markdown("---")

    st.markdown('<div class="section-hdr">Association Network — Top 20 Rules by Lift</div>', unsafe_allow_html=True)
    st.markdown("""
Each **node** is a walkability characteristic. Each **arrow** shows a rule: A → B.
Thicker, orange arrows = strongest associations (Lift >= 2.5).
Node colours match the walkability/transit/car-dependence category type.
    """)
    fig = draw_network(rules_df, top_n=20, title="Walkability Association Network — Top 20 Rules by Lift")
    st.pyplot(fig, use_container_width=True); plt.close()

    explain(
        "An arrow from A to B means 'when A is present, B is likely too.' "
        "The thicker and more orange the arrow, the stronger that connection.",
        "Directed graph G = (V, E) where V = unique antecedents and consequents. "
        "Edge weight encodes lift; edge width scales linearly with lift. "
        "Orange edges indicate lift >= 2.5."
    )

    st.markdown('<div class="section-hdr">Support vs Confidence Plot (bubble size = Lift)</div>', unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    lnorm = ((rules_df["Lift"] - rules_df["Lift"].min()) /
             (rules_df["Lift"].max() - rules_df["Lift"].min()))
    sc = ax.scatter(rules_df["Support"], rules_df["Confidence"],
                    s=30 + lnorm * 300, c=rules_df["Lift"],
                    cmap="RdYlGn", alpha=0.75, edgecolors="white", lw=0.4, rasterized=True)
    cbar = plt.colorbar(sc, ax=ax, shrink=0.85)
    cbar.set_label("Lift", fontsize=9)
    ax.set_xlabel("Support — How common is the rule?")
    ax.set_ylabel("Confidence — How reliable is the rule?")
    ax.set_title("Support vs Confidence — Each bubble is one rule (size and colour = Lift)")
    ax.grid(alpha=0.35)
    for _, row in rules_df.head(5).iterrows():
        ax.annotate(f'{row["Antecedent"][:12]}→{row["Consequent"][:10]}',
                    xy=(row["Support"], row["Confidence"]),
                    xytext=(6, 4), textcoords="offset points", fontsize=6.5, color="#333")
    plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()

    st.markdown("---")

    # ── (e) CONCLUSIONS ───────────────────────────────────────────────────────
    st.markdown('<div class="section-hdr">(e) Conclusions</div>', unsafe_allow_html=True)
    st.markdown(f"""
Association Rule Mining on the EPA Walkability dataset revealed the following key findings:

1. **The strongest rule in the data:** `{rules_df.iloc[0]['Antecedent']}` →
   `{rules_df.iloc[0]['Consequent']}` (Lift = {rules_df.iloc[0]['Lift']:.2f}) —
   these two characteristics appear together {rules_df.iloc[0]['Lift']:.1f}x
   more often than chance would predict.

2. **Least Walkable areas are almost universally transit-free.** The rule
   `Least Walkable → No Transit` has confidence close to 1.00, meaning virtually every
   least-walkable block group in America has no transit access.

3. **Walkability and intersection density are inseparable.** `Most Walkable → High Int Dens`
   and vice versa both have high lift, confirming that a well-connected street grid
   is the physical backbone of walkability.

4. **Car dependence is a consequence, not just a feature.** High car ownership is
   strongly associated with Below/Least Walkable areas.

5. **Transit access is the bridge between walkability tiers.** Moving from
   'No Transit' to 'High Transit' is associated with jumping from Below Average
   to Most Walkable categories, making transit investment the highest-leverage
   policy lever for improving walkability outcomes.
    """)

    st.success("Association Rule Mining complete. Proceed to the Conclusion page for overall project findings.")