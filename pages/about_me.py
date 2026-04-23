"""
About Me Page — EPA Walkability Index
Author: Sejal Hukare
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import warnings
from PIL import Image
import base64
from io import BytesIO
warnings.filterwarnings("ignore")

BLUE   = "#3498db"; GREEN  = "#2ecc71"; RED    = "#e74c3c"
ORANGE = "#f39c12"; PURPLE = "#9b59b6"; TEAL   = "#1abc9c"

CAT_ORDER = ["Least Walkable", "Below Average Walkable",
             "Above Average Walkable", "Most Walkable"]

CAT_COLORS = {
    "Least Walkable":           RED,
    "Below Average Walkable":   ORANGE,
    "Above Average Walkable":   TEAL,
    "Most Walkable":            GREEN,
}

CAT_DESCRIPTIONS = {
    "Least Walkable":
        "This area is highly car-dependent. Almost all errands require a car, "
        "transit is sparse or absent, and the street network is disconnected. "
        "This describes roughly 35% of all US neighborhoods.",
    "Below Average Walkable":
        "Some destinations are reachable on foot, but a car is still needed "
        "for most trips. Typical of outer suburbs and mid-density residential areas. "
        "About 30% of US neighborhoods fall here.",
    "Above Average Walkable":
        "Most errands can be accomplished on foot. Transit access is reasonable "
        "and the street grid is fairly connected. Inner suburbs and mid-size city "
        "neighborhoods often land in this tier.",
    "Most Walkable":
        "Daily errands do not require a car. Dense street networks, excellent "
        "transit access, and diverse land uses make walking the natural choice. "
        "This describes the urban cores of America's most walkable cities.",
}


@st.cache_data(show_spinner=False)
def load_predictor():
    """Train a Random Forest on the full dataset for live predictions."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    df = pd.read_csv("walkability_cleaned.csv")
    df["D4A_clean"] = df["D4A"].replace(-99999, 0)
    df = df[df["Walkability_Category"].notna()]

    features = ["D2A_Ranked", "D2B_Ranked", "D3B_Ranked", "D4A_Ranked",
                "NatWalkInd", "D3B", "D4A_clean", "D2B_E8MIXA",
                "Pct_AO0", "Pct_AO1", "Pct_AO2p"]

    df_model = df[features + ["Walkability_Category"]].dropna()
    X = df_model[features]
    y = df_model["Walkability_Category"]

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_tr, y_tr)
    acc = accuracy_score(y_te, rf.predict(X_te))

    # Feature stats for slider defaults
    stats = df_model[features].describe()

    return rf, features, acc, stats


def app():
    st.markdown("""
    <style>
        #MainMenu {visibility: hidden;}
        footer    {visibility: hidden;}

        .profile-name {
            font-size: 42px; font-weight: 800; color: #f1f5f9;
            letter-spacing: -0.5px; margin-bottom: 4px;
        }
        .profile-title {
            font-size: 16px; color: #3498db; font-weight: 600;
            letter-spacing: 1px; text-transform: uppercase;
            margin-bottom: 16px;
        }
        .profile-bio {
            font-size: 15px; color: #94a3b8; line-height: 1.85;
            margin-bottom: 20px;
        }
        .link-btn {
            display: inline-block; padding: 8px 18px; border-radius: 8px;
            font-size: 13px; font-weight: 600; text-decoration: none;
            margin-right: 10px; margin-bottom: 8px; border: 1px solid;
        }
        .stat-card {
            background: #1e293b; border: 1px solid #334155;
            border-radius: 12px; padding: 20px 16px; text-align: center;
        }
        .stat-val {
            font-size: 32px; font-weight: 800; line-height: 1; margin-bottom: 6px;
        }
        .stat-lbl {
            font-size: 11px; color: #64748b; text-transform: uppercase;
            letter-spacing: 1px; font-weight: 600;
        }
        .pred-result {
            border-radius: 14px; padding: 24px 28px; margin-top: 20px;
        }
        .pred-tier {
            font-size: 28px; font-weight: 800; margin-bottom: 6px;
        }
        .pred-desc {
            font-size: 14px; color: #cbd5e1; line-height: 1.7;
        }
        .soft-divider {
            border: none; border-top: 1px solid #1e293b; margin: 44px 0;
        }
        .section-label {
            font-size: 11px; font-weight: 700; letter-spacing: 2.5px;
            text-transform: uppercase; color: #3498db; margin-bottom: 8px;
        }
        .section-heading {
            font-size: 26px; font-weight: 700; color: #f1f5f9;
            margin-bottom: 20px; line-height: 1.3;
        }
        .skill-pill {
            display: inline-block; padding: 5px 14px; border-radius: 999px;
            font-size: 12px; font-weight: 600; margin: 4px;
            background: #1e293b; border: 1px solid #334155; color: #94a3b8;
        }
    </style>
    """, unsafe_allow_html=True)

    # ── PROFILE SECTION ───────────────────────────────────────────────────
    col_img, col_bio = st.columns([1, 2.5], gap="large")

    with col_img:
        # Profile image — place profile_walk.jpg/png in project root
        img_path = None
        for ext in ["jpg", "jpeg", "png", "webp"]:
            candidate = f"profile_walk.{ext}"
            if os.path.exists(candidate):
                img_path = candidate
                break

        if img_path:
            img = Image.open(img_path).convert("RGBA")

            w, h = img.size
            side = min(w, h)
            left = (w - side) // 2
            top  = (h - side) // 2
            img  = img.crop((left, top, left + side, top + side))

            buf = BytesIO()
            img.save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode()

            st.markdown(f"""
            <div style="display:flex;justify-content:center;align-items:center;">
                <img src="data:image/png;base64,{b64}"
                    style="width:240px;height:240px;border-radius:50%;
                        object-fit:cover;border:3px solid #3498db;" />
            </div>
            """, unsafe_allow_html=True)

            
        else:
            # Placeholder if image not found
            st.markdown("""
            <div style="background:#1e293b;border:2px dashed #334155;border-radius:16px;
            height:280px;display:flex;align-items:center;justify-content:center;
            color:#475569;font-size:13px;text-align:center;padding:20px;">
            Add profile_walk.jpg<br>to project root
            </div>
            """, unsafe_allow_html=True)

    with col_bio:
        st.markdown('<div class="profile-name">Sejal Hukare</div>', unsafe_allow_html=True)
        st.markdown('<div class="profile-title">Data Science &nbsp;·&nbsp; CU Boulder</div>', unsafe_allow_html=True)

        st.markdown("""
        <p class="profile-bio">
        I built this project to explore one of the most tangible ways data science
        connects to everyday life — the design of the neighborhoods we live in.
        Using the EPA's National Walkability Index across 220,000+ US Census block
        groups, I applied eight machine learning methods to understand what drives
        walkability and how accurately we can predict it.
        </p>
        <p class="profile-bio">
        This app is the result — an end-to-end data science pipeline from raw data
        cleaning through PCA, clustering, association rule mining, supervised learning,
        SVMs, and ensemble methods, all built in Python and deployed with Streamlit.
        </p>
        """, unsafe_allow_html=True)

        # Links
        st.markdown("""
        <a class="link-btn" href="https://www.linkedin.com/in/sejal-hukare/"
           target="_blank"
           style="color:#0077b5;border-color:#0077b5;background:#0077b522;">
           LinkedIn
        </a>
        <a class="link-btn" href="https://github.com/sezol"
           target="_blank"
           style="color:#f1f5f9;border-color:#334155;background:#1e293b;">
           GitHub
        </a>
        <a class="link-btn" href="https://www.researchgate.net/profile/Sejal-Hukare"
           target="_blank"
           style="color:#00d2b5;border-color:#00d2b5;background:#00d2b522;">
           ResearchGate
        </a>
        <a class="link-btn" href="https://github.com/sezol/my_walkability_app"
           target="_blank"
           style="color:#f39c12;border-color:#f39c12;background:#f39c1222;">
           Project Repo
        </a>
        """, unsafe_allow_html=True)

        # Skills
        skills = ["Python", "Streamlit", "Scikit-learn", "Pandas", "Matplotlib",
                  "Random Forest", "SVM", "PCA", "Clustering", "EPA NWI",
                  "Urban Analytics", "Machine Learning"]
        pills = "".join([f'<span class="skill-pill">{s}</span>' for s in skills])
        st.markdown(f'<div style="margin-top:12px;">{pills}</div>', unsafe_allow_html=True)

    st.markdown('<hr class="soft-divider">', unsafe_allow_html=True)

    # ── PROJECT STATS ─────────────────────────────────────────────────────
    st.markdown('<p class="section-label">By The Numbers</p>', unsafe_allow_html=True)

    stats_data = [
        ("220K+",  "Census Block Groups",    BLUE),
        ("50",     "US States Covered",      GREEN),
        ("8",      "ML Methods Used",        PURPLE),
        ("86%+",   "Best Model Accuracy",    ORANGE),
        ("43K",    "Training Neighborhoods", TEAL),
        ("70,123", "Health Study Tracts",    RED),
    ]

    cols = st.columns(6)
    for col, (val, label, color) in zip(cols, stats_data):
        with col:
            st.markdown(f"""
            <div class="stat-card" style="border-top:3px solid {color};">
                <div class="stat-val" style="color:{color};">{val}</div>
                <div class="stat-lbl">{label}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<hr class="soft-divider">', unsafe_allow_html=True)

    # ── LIVE WALKABILITY PREDICTOR ────────────────────────────────────────
    st.markdown('<p class="section-label">Try It Live</p>', unsafe_allow_html=True)
    st.markdown('<h2 class="section-heading">Predict Your Neighborhood\'s Walkability</h2>', unsafe_allow_html=True)

    st.markdown("""
    <p style="font-size:15px;color:#94a3b8;line-height:1.8;margin-bottom:24px;">
    The Random Forest model trained on 43,000+ US neighborhoods is running live below.
    Adjust the sliders to match your area's characteristics and see which walkability
    tier the model predicts — instantly, using the same algorithm that achieved 86%+ accuracy.
    </p>
    """, unsafe_allow_html=True)

    rf, features, model_acc, feat_stats = load_predictor()

    col_sliders, col_result = st.columns([1.6, 1], gap="large")

    with col_sliders:
        st.markdown("**Street & Transit**")
        c1, c2 = st.columns(2)
        with c1:
            d3b_ranked = st.slider("Intersection Density (Ranked 1–20)",
                                   1, 20, 10, key="d3b_r",
                                   help="How connected is the street grid? Higher = more intersections per km²")
            d4a_ranked = st.slider("Transit Proximity (Ranked 1–20)",
                                   1, 20, 8, key="d4a_r",
                                   help="How close is the nearest bus/train stop? Higher = closer")
        with c2:
            d3b_raw    = st.slider("Intersection Density (raw, /km²)",
                                   0, 300, 60, key="d3b_raw",
                                   help="Actual intersection density per square kilometer")
            d4a_clean  = st.slider("Transit Distance (metres, 0=none)",
                                   0, 2000, 400, key="d4a_clean",
                                   help="0 = no transit access. Lower = transit is closer")

        st.markdown("**Land Use & Employment**")
        c3, c4 = st.columns(2)
        with c3:
            d2a_ranked = st.slider("Emp-HH Mix (Ranked 1–20)",
                                   1, 20, 8, key="d2a_r",
                                   help="Balance of jobs and households nearby")
            d2b_ranked = st.slider("Employment Mix (Ranked 1–20)",
                                   1, 20, 8, key="d2b_r",
                                   help="Diversity of nearby job types")
        with c4:
            d2b_mix    = st.slider("Employment Mix (0–1 entropy)",
                                   0.0, 1.0, 0.4, step=0.01, key="d2b_mix",
                                   help="0 = single land use, 1 = fully mixed uses")
            nat_walk   = st.slider("National Walkability Index (1–20)",
                                   1.0, 20.0, 10.0, step=0.5, key="nwi",
                                   help="The EPA composite score combining all four components")

        st.markdown("**Car Ownership**")
        c5, c6, c7 = st.columns(3)
        with c5:
            pct_ao0 = st.slider("% Zero-car households",
                                0.0, 1.0, 0.1, step=0.01, key="ao0",
                                help="Fraction of households with no car")
        with c6:
            pct_ao1 = st.slider("% One-car households",
                                0.0, 1.0, 0.4, step=0.01, key="ao1",
                                help="Fraction of households with exactly one car")
        with c7:
            pct_ao2 = st.slider("% Two+ car households",
                                0.0, 1.0, 0.5, step=0.01, key="ao2",
                                help="Fraction of households with two or more cars")

        # D2A EPHHM — derived estimate
        d2a_ephhm = d2b_mix * 0.8 + (1 - pct_ao2) * 0.2

    with col_result:
        # Build input vector
        input_vec = pd.DataFrame([[
            d2a_ranked, d2b_ranked, d3b_ranked, d4a_ranked,
            nat_walk, d3b_raw, d4a_clean, d2b_mix,
            pct_ao0, pct_ao1, pct_ao2
        ]], columns=features)

        prediction  = rf.predict(input_vec)[0]
        proba       = rf.predict_proba(input_vec)[0]
        proba_dict  = dict(zip(rf.classes_, proba))
        color       = CAT_COLORS.get(prediction, BLUE)
        description = CAT_DESCRIPTIONS.get(prediction, "")

        st.markdown(f"""
        <div class="pred-result" style="background:{color}18;border:2px solid {color};">
            <div style="font-size:11px;font-weight:700;letter-spacing:2px;
            color:{color};text-transform:uppercase;margin-bottom:8px;">
            Model Prediction
            </div>
            <div class="pred-tier" style="color:{color};">{prediction}</div>
            <div class="pred-desc">{description}</div>
        </div>
        """, unsafe_allow_html=True)

        # Confidence bars
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("**Confidence per tier:**")
        for cat in CAT_ORDER:
            p    = proba_dict.get(cat, 0)
            c    = CAT_COLORS.get(cat, BLUE)
            bold = "font-weight:700;" if cat == prediction else ""
            st.markdown(f"""
            <div style="margin-bottom:8px;">
                <div style="font-size:12px;color:#94a3b8;{bold}margin-bottom:3px;">
                    {cat}
                </div>
                <div style="background:#1e293b;border-radius:4px;height:8px;">
                    <div style="background:{c};width:{p*100:.1f}%;height:8px;
                    border-radius:4px;transition:width 0.3s;"></div>
                </div>
                <div style="font-size:11px;color:{c};font-weight:600;margin-top:2px;">
                    {p*100:.1f}%
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown(f"""
        <div style="margin-top:16px;background:#1e293b;border-radius:8px;
        padding:12px 16px;font-size:12px;color:#64748b;line-height:1.6;">
        Model: Random Forest (100 trees)<br>
        Training accuracy: {model_acc*100:.1f}% on 43K+ neighborhoods<br>
        Data: EPA National Walkability Index 2021
        </div>
        """, unsafe_allow_html=True)

    # ── Example neighborhoods ─────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("**Try these real-world presets:**")

    preset_cols = st.columns(4)
    presets = [
        ("Manhattan, NY",        19, 20, 18, 20, 19.5, 180, 50,  0.8, 0.4, 0.15, 0.05),
        ("Denver Suburbs, CO",   8,  6,  9,  7,  9.0,  40, 600, 0.3, 0.08, 0.45, 0.47),
        ("Chicago Loop, IL",     17, 18, 16, 19, 17.8, 140, 80,  0.65, 0.35, 0.35, 0.20),
        ("Rural Kansas",         2,  1,  2,  1,  2.5,  8, 0,   0.05, 0.01, 0.30, 0.69),
    ]

    for col, (name, d2a_r, d2b_r, d3b_r, d4a_r, nwi,
              d3b_rv, d4a_cv, ao0, ao1, ao2, _) in zip(preset_cols, presets):
        with col:
            if st.button(name, use_container_width=True, key=f"preset_{name}"):
                sample = pd.DataFrame([[d2a_r, d2b_r, d3b_r, d4a_r, nwi,
                                         d3b_rv, d4a_cv, 0.5,
                                         ao0, ao1, ao2]],
                                       columns=features)
                pred  = rf.predict(sample)[0]
                prob  = rf.predict_proba(sample)[0]
                color = CAT_COLORS.get(pred, BLUE)
                st.markdown(f"""
                <div style="background:{color}18;border:1px solid {color};
                border-radius:8px;padding:10px;margin-top:6px;text-align:center;">
                    <div style="font-size:11px;color:{color};font-weight:700;">
                    {pred}
                    </div>
                    <div style="font-size:10px;color:#64748b;margin-top:3px;">
                    {max(prob)*100:.0f}% confidence
                    </div>
                </div>
                """, unsafe_allow_html=True)

    st.markdown('<hr class="soft-divider">', unsafe_allow_html=True)

    # ── ABOUT THE PROJECT ─────────────────────────────────────────────────
    st.markdown('<p class="section-label">About This Project</p>', unsafe_allow_html=True)
    st.markdown('<h2 class="section-heading">What Was Built Here</h2>', unsafe_allow_html=True)

    method_cols = st.columns(4, gap="large")
    methods = [
        ("Exploratory Analysis",
         "PCA, K-Means, DBSCAN, and hierarchical clustering revealed that the four EPA "
         "walkability tiers are genuinely distinct — not arbitrary score thresholds.",
         BLUE),
        ("Pattern Mining",
         "Association rule mining found near-universal laws: no transit access predicts "
         "least walkable neighborhoods with ~100% confidence across all 43K block groups.",
         ORANGE),
        ("Supervised Learning",
         "Naive Bayes, Decision Trees, and Logistic Regression established that walkability "
         "can be predicted from features alone with 72–80% accuracy.",
         PURPLE),
        ("Advanced Models",
         "SVM (RBF kernel) and Random Forest pushed accuracy above 83% and 86% respectively, "
         "confirming non-linear class boundaries in the feature space.",
         GREEN),
    ]

    for col, (title, body, color) in zip(method_cols, methods):
        with col:
            st.markdown(f"""
            <div style="background:#1e293b;border:1px solid #334155;border-top:3px solid {color};
            border-radius:10px;padding:18px;height:100%;">
                <div style="font-size:13px;font-weight:700;color:#f1f5f9;margin-bottom:8px;">
                {title}
                </div>
                <div style="font-size:12px;color:#64748b;line-height:1.7;">{body}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<div style="margin-bottom:80px;"></div>', unsafe_allow_html=True)