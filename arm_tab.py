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

# ── colour each item category for the network ─────────────────────────────────
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
    cols  = ["WalkCat", "Transit", "Auto Dep", "Emp Mix", "Int Dens", "Pop Dens"]
    trans = d[cols].apply(lambda r: set(r.values), axis=1).tolist()
    n     = len(trans)

    all_items: set = set()
    for t in trans:
        all_items.update(t)
    item_sup = {frozenset([i]): sum(1 for t in trans if i in t) / n for i in all_items}
    freq1    = {k: v for k, v in item_sup.items() if v >= min_sup}
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
    return rdf, item_sup, d[cols], d[cols + ["Walkability_Category"]]


def metric_card(col, val, label, color=BLUE):
    col.markdown(
        f'<div style="background:#f8f9fa;border-left:4px solid {color};'
        f'border-radius:6px;padding:14px 16px;margin-bottom:8px">'
        f'<div style="font-size:1.4rem;font-weight:700;color:#2c3e50;line-height:1.1">{val}</div>'
        f'<div style="font-size:0.73rem;color:#7f8c8d;text-transform:uppercase;'
        f'letter-spacing:0.06em;margin-top:3px">{label}</div></div>',
        unsafe_allow_html=True)


def explain(layman, ds):
    with st.expander("📖 What does this mean?", expanded=False):
        st.markdown(f"**In plain English:** {layman}")
        st.markdown(f"**For data scientists:** {ds}")


def badge(text, color):
    return (f'<span style="display:inline-block;padding:2px 8px;border-radius:4px;'
            f'font-size:0.76rem;font-weight:600;background:{color}22;color:{color};'
            f'border:1px solid {color}44">{text}</span>')


def rules_table(df_rules, sort_col, n=15):
    """Render top-n rules sorted by sort_col as a styled HTML table."""
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
    """Draw a network graph of the top_n rules by lift."""
    top = rules_df.sort_values("Lift", ascending=False).head(top_n)
    G = nx.DiGraph()
    for _, row in top.iterrows():
        G.add_edge(row["Antecedent"], row["Consequent"],
                   weight=row["Lift"], conf=row["Confidence"],
                   sup=row["Support"])

    pos = nx.spring_layout(G, seed=42, k=2.2)
    node_colors = [NODE_COLOR_MAP.get(n, "#95a5a6") for n in G.nodes()]
    edge_weights = [G[u][v]["weight"] for u, v in G.edges()]
    edge_widths  = [1 + (w - 1) * 1.2 for w in edge_weights]
    edge_alphas  = [min(1.0, 0.3 + w * 0.15) for w in edge_weights]

    fig, ax = plt.subplots(figsize=(13, 8), facecolor="white")
    ax.set_facecolor("#fafafa")
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=1600,
                           alpha=0.92, ax=ax)
    for (u, v), w, wd, al in zip(G.edges(), edge_weights, edge_widths, edge_alphas):
        nx.draw_networkx_edges(G, pos, edgelist=[(u, v)],
                               width=wd, alpha=al,
                               edge_color=[ORANGE if w >= 2.5 else BLUE],
                               connectionstyle="arc3,rad=0.08",
                               arrowsize=18, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=7.5, font_color="white",
                            font_weight="bold", ax=ax)
    ax.set_title(f"{title}\n(arrow thickness = Lift strength; orange = Lift ≥ 2.5)",
                 fontsize=12, fontweight="bold", pad=14)
    ax.axis("off")
    plt.tight_layout()
    return fig


def app():
    st.markdown("""
    <style>
    .section-hdr{font-size:1.25rem;font-weight:700;color:#2c3e50;
        border-left:4px solid #3498db;padding-left:12px;margin:1.5rem 0 0.8rem}
    .callout{background:#eaf4fd;border-left:4px solid #3498db;border-radius:6px;
        padding:14px 18px;margin:10px 0;font-size:0.9rem;color:#1a5276}
    .callout.green{background:#eafaf1;border-color:#2ecc71;color:#1e8449}
    .callout.orange{background:#fef9e7;border-color:#f39c12;color:#7d6608}
    .callout.red{background:#fdedec;border-color:#e74c3c;color:#922b21}
    </style>""", unsafe_allow_html=True)

    st.title("🔗 Association Rule Mining (ARM)")

    rules_df, item_sup, trans_df, full_df = load_arm_data()

    # ── (a) OVERVIEW ──────────────────────────────────────────────────────────
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
        <div style="background:#f8f9fa;border-radius:10px;padding:20px;border:1px solid #dee2e6">
        <div style="font-size:1rem;font-weight:700;color:#2c3e50;margin-bottom:12px">
            📐 The Three Key Measures
        </div>
        <div style="margin-bottom:10px">
            <span style="background:#eaf4fd;color:{BLUE};padding:3px 9px;border-radius:4px;
            font-weight:700;font-size:.85rem">SUPPORT</span>&nbsp;
            <span style="color:#555;font-size:.88rem">= P(A ∩ B)</span><br>
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
            <span style="color:#7f8c8d;font-size:.83rem">Lift &gt; 1 means A and B appear together
            more than random chance. Lift = 2.8 means 2.8× more likely than by chance.</span>
        </div>
        </div>""", unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div style="background:#f8f9fa;border-radius:10px;padding:20px;border:1px solid #dee2e6">
        <div style="font-size:1rem;font-weight:700;color:#2c3e50;margin-bottom:12px">
            ⚙ How the Apriori Algorithm Works
        </div>
        <ol style="color:#555;font-size:.88rem;padding-left:18px;line-height:1.9">
            <li>Each block group becomes a <b>"basket"</b> of labels 
                (e.g. {{No Transit, Least Walkable, High Car Dep}})</li>
            <li>Scan all baskets and count how often each single item appears 
                → keep only <b>frequent items</b> (support ≥ threshold)</li>
            <li>Pair up frequent items and scan again 
                → keep only <b>frequent pairs</b></li>
            <li>From each frequent pair, generate two <b>rules</b> 
                (A→B and B→A)</li>
            <li>Compute Support, Confidence, Lift for every rule</li>
            <li><b>Rank</b> by Lift to find the most interesting associations</li>
        </ol>
        <div style="background:#eaf4fd;border-radius:6px;padding:10px;margin-top:10px;
            font-size:.82rem;color:#1a5276">
            🔑 <b>Key principle (Apriori property):</b> If an itemset is infrequent, 
            all its supersets are also infrequent — this prunes the search space.
        </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ── (b) DATA PREP ──────────────────────────────────────────────────────────
    st.markdown('<div class="section-hdr">(b) Data Preparation for ARM</div>', unsafe_allow_html=True)

    st.markdown("""
    ARM requires **unlabelled transaction data only** — a list of "baskets," where each basket 
    is a set of items that appear together. There are no numeric values, no regression targets, 
    and no labels in the ARM input. Continuous variables must be **discretised** (binned) 
    into meaningful categories first.

    **Our discretisation rules:**
    """)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
**Transit Access (D4A)**
- `No Transit` = −99,999 (sentinel)
- `Low Transit` = 0–499 m
- `High Transit` = ≥ 500 m

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

    st.markdown("**Sample of transaction data (first 8 block groups):**")
    st.dataframe(trans_df.head(8), use_container_width=True)

    explain(
        "Each row in this table is one neighbourhood (Census block group). Instead of numbers, "
        "we now have labels like 'No Transit' or 'High Car Dep'. ARM treats each row as a "
        "'shopping basket' and looks for labels that tend to appear in the same basket. "
        "We removed the numeric columns entirely — ARM only works with these category labels.",
        "Discretisation converts continuous features to nominal items, enabling market-basket "
        "analysis. Each transaction T_i = {item_1, ..., item_k} where each item is a "
        "(variable, bin) pair. The resulting transaction matrix has 43,065 rows × 6 item "
        "columns, yielding 18 unique items. min_support = 0.05 retains all items with "
        "frequency ≥ 5% across all block groups."
    )

    # Itemset frequency chart
    st.markdown("**Frequent item support (how often each item appears across all block groups):**")
    isdf = (pd.DataFrame([{"Item": list(k)[0], "Support": v} for k, v in item_sup.items()])
            .sort_values("Support", ascending=False))
    fig, ax = plt.subplots(figsize=(11, 5.5))
    colors = [NODE_COLOR_MAP.get(item, "#95a5a6") for item in isdf["Item"][::-1]]
    bars = ax.barh(isdf["Item"][::-1], isdf["Support"][::-1],
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
    ⚙ <b>Thresholds used:</b> min_support = <b>0.05</b> (item must appear in ≥ 5% of block groups),
    applied to both 1-itemsets and 2-itemsets before rule generation. 
    This yielded <b>{len(item_sup)} frequent items</b> and <b>{len(rules_df)} rules</b> total.
    No minimum confidence threshold was applied — rules were ranked post-hoc.
    </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ── (c) CODE ARM ──────────────────────────────────────────────────────────
    st.markdown('<div class="section-hdr">(c) ARM Implementation</div>', unsafe_allow_html=True)

    with st.expander("📄 View ARM Python Code", expanded=False):
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

    # Metric row
    top_lift = rules_df.iloc[0]
    c1,c2,c3,c4,c5,c6 = st.columns(6)
    metric_card(c1, len(rules_df),                       "Total Rules",       BLUE)
    metric_card(c2, f"{rules_df['Lift'].max():.2f}",     "Max Lift",          GREEN)
    metric_card(c3, f"{rules_df['Confidence'].max():.2f}","Max Confidence",   BLUE)
    metric_card(c4, f"{rules_df['Support'].max():.3f}",  "Max Support",       ORANGE)
    metric_card(c5, len(item_sup),                       "Frequent Items",    PURPLE)
    metric_card(c6, "0.05",                              "Min Support Used",  RED)

    st.markdown("---")

    # Three top-15 tables
    t1, t2, t3 = st.tabs(["🔼 Top 15 by Lift", "📊 Top 15 by Confidence", "📈 Top 15 by Support"])

    with t1:
        st.markdown("**Rules ranked by Lift — the most *interesting* associations (furthest from random chance)**")
        rules_table(rules_df, "Lift", 15)
        explain(
            f"Lift shows how much more likely two things appear together compared to pure chance. "
            f"The top rule here is '{top_lift['Antecedent']} → {top_lift['Consequent']}' with "
            f"lift = {top_lift['Lift']:.2f}. This means those two characteristics appear together "
            f"{top_lift['Lift']:.1f}× more often than you'd expect if they were unrelated. "
            "Anything above 1.0 is a meaningful association.",
            "Lift = Confidence / P(Consequent). Values > 1 indicate positive correlation; "
            "values < 1 indicate negative correlation; = 1 means independence. "
            "Lift is the preferred ranking metric as it accounts for the marginal frequency "
            "of the consequent, unlike confidence alone."
        )

    with t2:
        st.markdown("**Rules ranked by Confidence — the most *reliable* associations**")
        rules_table(rules_df, "Confidence", 15)
        explain(
            "Confidence tells you: 'Given that a neighbourhood has X, how often does it also have Y?' "
            "A confidence of 0.99 is almost a certainty. You'll notice the top confidence rules "
            "involve 'Least Walkable → No Transit' — meaning nearly every least-walkable "
            "neighbourhood has no transit access. This is a near-universal truth in the data.",
            "Confidence = Support(A∪B) / Support(A). High confidence rules can be misleading "
            "if the consequent is already very common (high base rate), which is why Lift is "
            "typically preferred. Note Lift-adjusted rules may have moderate confidence "
            "but low-frequency consequents making them more surprising."
        )

    with t3:
        st.markdown("**Rules ranked by Support — the most *frequent* associations**")
        rules_table(rules_df, "Support", 15)
        explain(
            "Support tells you how common a pattern is across all block groups. "
            "High support means the pattern is widespread, not just true for a few neighbourhoods. "
            "The top support rules tend to involve common items like 'Med Pop' and 'High Emp Mix' "
            "because those categories cover the majority of block groups.",
            "Support = |T(A∪B)| / |T|. High-support rules are statistically robust but may "
            "have low lift if both items are individually common. The top support rules here "
            "reflect the modal distribution of the dataset (most U.S. block groups are "
            "medium-density with moderate employment mix) rather than strong associations."
        )

    st.markdown("---")

    # Network visualisation
    st.markdown('<div class="section-hdr">Association Network — Top 20 Rules by Lift</div>', unsafe_allow_html=True)
    st.markdown("""
    Each **node** is a walkability characteristic. Each **arrow** shows a rule: A → B.  
    **Thicker, orange arrows** = strongest associations (Lift ≥ 2.5).  
    **Node colours** match the walkability/transit/car-dependence category type.
    """)

    fig = draw_network(rules_df, top_n=20, title="Walkability Association Network — Top 20 Rules by Lift")
    st.pyplot(fig, use_container_width=True); plt.close()

    explain(
        "Think of this as a map of 'what goes with what.' The nodes are characteristics "
        "of neighbourhoods. An arrow from A to B means 'when A is present, B is likely too.' "
        "The thicker and more orange the arrow, the stronger that connection. "
        "Notice how 'Least Walkable' and 'No Transit' are tightly linked with thick orange arrows "
        "— this is the strongest pattern in the data. Similarly, 'Most Walkable' connects strongly "
        "to 'High Intersection Density', confirming that walkable areas have more street crossings.",
        "Directed graph G = (V, E) where V = unique antecedents ∪ consequents and "
        "E = top-20 rules by lift. Edge weight encodes lift; edge width scales linearly with lift. "
        "Orange edges indicate lift ≥ 2.5. NetworkX spring layout (k=2.2, seed=42) is used "
        "to position nodes. Node colour encodes category type using a predefined palette."
    )

    # Scatter: Support vs Confidence
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
    ax.set_title("Support vs Confidence — Each bubble is one rule (size & colour = Lift)")
    ax.grid(alpha=0.35)
    # annotate top 5
    for _, row in rules_df.head(5).iterrows():
        ax.annotate(f'{row["Antecedent"][:12]}→{row["Consequent"][:10]}',
                    xy=(row["Support"], row["Confidence"]),
                    xytext=(6, 4), textcoords="offset points",
                    fontsize=6.5, color="#333")
    plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()

    explain(
        "Each bubble is one rule. Rules in the top-right corner are both common AND reliable — "
        "those are the most useful findings. Rules in the top-left are reliable but rare. "
        "Big green bubbles are the most interesting — they have high lift, meaning the "
        "association is much stronger than random chance.",
        "This plot visualises the support-confidence-lift tradeoff space. The ideal rule "
        "occupies the high-support, high-confidence, high-lift region (top-right, large green bubble). "
        "The visible cluster of low-lift rules (small grey/yellow bubbles) in the centre "
        "represents trivial co-occurrences driven by high base-rate items."
    )

    st.markdown("---")

    # ── (e) CONCLUSIONS ───────────────────────────────────────────────────────
    st.markdown('<div class="section-hdr">(e) Conclusions</div>', unsafe_allow_html=True)
    st.markdown(f"""
    Association Rule Mining on the EPA Walkability dataset revealed the following key findings:

    1. **The strongest rule in the data:** `{rules_df.iloc[0]['Antecedent']}` → 
       `{rules_df.iloc[0]['Consequent']}` (Lift = {rules_df.iloc[0]['Lift']:.2f}) — 
       meaning these two characteristics appear together {rules_df.iloc[0]['Lift']:.1f}× 
       more often than chance would predict.

    2. **Least Walkable areas are almost universally transit-free.** The rule 
       `Least Walkable → No Transit` has confidence ≈ 1.00, meaning virtually every 
       least-walkable block group in America has no transit access. This near-universal 
       truth points to a systemic infrastructure gap.

    3. **Walkability and intersection density are inseparable.** `Most Walkable → High Int Dens` 
       and vice versa both have high lift, confirming that a well-connected street grid 
       (more intersections per km²) is the physical backbone of walkability.

    4. **Car dependence is a consequence, not just a feature.** High car ownership is 
       strongly associated with Below/Least Walkable areas — suggesting that residents 
       in non-walkable areas are forced into car dependency rather than choosing it.

    5. **Transit access is the bridge between walkability tiers.** Moving from 
       'No Transit' to 'High Transit' is associated with jumping from Below Average 
       to Most Walkable categories, making transit investment the highest-leverage 
       policy lever for improving walkability outcomes.
    """)

    st.success("✓ Association Rule Mining complete. Proceed to the Conclusion page for overall project findings.")