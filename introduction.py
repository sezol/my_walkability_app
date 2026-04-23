import streamlit as st
import streamlit.components.v1 as components

def app():
    st.markdown("""
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}

        /* ── Typography ── */
        .hero-title {
            font-size: 72px;
            font-weight: 800;
            color: #ffffff;
            text-align: center;
            letter-spacing: -1px;
            line-height: 1.1;
            margin-bottom: 12px;
        }
        .hero-sub {
            font-size: 17px;
            color: #94a3b8;
            text-align: center;
            font-weight: 400;
            letter-spacing: 0.3px;
            margin-bottom: 48px;
        }
        .section-label {
            font-size: 11px;
            font-weight: 700;
            letter-spacing: 2.5px;
            text-transform: uppercase;
            color: #3498db;
            margin-bottom: 10px;
        }
        .section-heading {
            font-size: 26px;
            font-weight: 700;
            color: #f1f5f9;
            margin-bottom: 20px;
            line-height: 1.3;
        }
        .body-text {
            font-size: 16px;
            line-height: 1.95;
            color: #cbd5e1;
            margin-bottom: 0px;
        }

        /* ── Stat cards ── */
        .stat-row {
            display: flex;
            gap: 16px;
            margin: 36px 0;
            flex-wrap: wrap;
        }
        .stat-card {
            flex: 1;
            min-width: 140px;
            background: #1e293b;
            border: 1px solid #334155;
            border-radius: 12px;
            padding: 22px 20px;
            text-align: center;
        }
        .stat-value {
            font-size: 32px;
            font-weight: 800;
            color: #3498db;
            line-height: 1;
            margin-bottom: 6px;
        }
        .stat-label {
            font-size: 12px;
            color: #94a3b8;
            text-transform: uppercase;
            letter-spacing: 1px;
            font-weight: 500;
        }

        /* ── Pull quote ── */
        .pull-quote {
            border-left: 4px solid #3498db;
            padding: 18px 24px;
            background: #1e293b;
            border-radius: 0 10px 10px 0;
            margin: 32px 0;
            font-size: 17px;
            font-style: italic;
            color: #e2e8f0;
            line-height: 1.7;
        }

        /* ── Image card ── */
        .img-caption {
            font-size: 13px;
            color: #64748b;
            text-align: center;
            font-style: italic;
            margin-top: 10px;
            margin-bottom: 36px;
        }

        /* ── Topic tags ── */
        .tag-row {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            margin: 28px 0;
        }
        .tag {
            background: #1e293b;
            border: 1px solid #334155;
            color: #94a3b8;
            font-size: 13px;
            padding: 6px 14px;
            border-radius: 999px;
            font-weight: 500;
        }

        /* ── Divider ── */
        .soft-divider {
            border: none;
            border-top: 1px solid #1e293b;
            margin: 44px 0;
        }

        /* ── Research questions ── */
        .q-card {
            background: #1e293b;
            border: 1px solid #334155;
            border-radius: 10px;
            padding: 16px 20px;
            margin-bottom: 12px;
            display: flex;
            align-items: flex-start;
            gap: 14px;
        }
        .q-num {
            font-size: 13px;
            font-weight: 700;
            color: #3498db;
            min-width: 24px;
            padding-top: 1px;
        }
        .q-text {
            font-size: 15px;
            color: #cbd5e1;
            line-height: 1.6;
        }
    </style>
    """, unsafe_allow_html=True)

    # ── Hero ─────────────────────────────────────────────────────────────
    st.markdown('<div style="padding: 40px 0 20px 0;">', unsafe_allow_html=True)
    st.markdown('<h1 class="hero-title">Walkability Analysis</h1>', unsafe_allow_html=True)
    st.markdown('<p class="hero-sub">Understanding How Urban Design Shapes Movement in American Cities</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Stat cards ───────────────────────────────────────────────────────
    st.markdown("""
    <div class="stat-row">
        <div class="stat-card">
            <div class="stat-value">220K+</div>
            <div class="stat-label">Census Block Groups</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">50</div>
            <div class="stat-label">US States Covered</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">1–20</div>
            <div class="stat-label">Walkability Score Range</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">4</div>
            <div class="stat-label">Walkability Tiers</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">EPA</div>
            <div class="stat-label">Data Source</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<hr class="soft-divider">', unsafe_allow_html=True)

    # ── Section 1: What is Walkability ───────────────────────────────────
    st.markdown('<p class="section-label">Background</p>', unsafe_allow_html=True)
    st.markdown('<h2 class="section-heading">What is Walkability?</h2>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1], gap="large")
    with col1:
        st.markdown("""
        <p class="body-text">
        Walkability refers to how easily people can move through their neighborhoods on foot to reach
        everyday destinations such as grocery stores, schools, workplaces, parks, and healthcare services.
        In walkable environments, streets are designed to support pedestrians through connected road
        networks, safe crossings, and a mix of land uses that place essential services within reasonable
        walking distances.
        </p>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <p class="body-text">
        These environments encourage daily physical activity, reduce dependence on automobiles, and
        promote stronger social interactions among residents. Walkability has become an increasingly
        important concept as cities grow denser and communities seek more sustainable forms of mobility.
        Neighborhoods that support walking tend to foster healthier lifestyles, improved air quality,
        and greater economic vitality.
        </p>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="pull-quote">
    "The presence of sidewalks, intersections, and nearby destinations shapes how people experience
    their surroundings and interact with their community — and walkability influences how accessible
    a city feels to everyone, including children, older adults, and people with disabilities."
    </div>
    """, unsafe_allow_html=True)

    # ── Image 1 ───────────────────────────────────────────────────────────
    st.image(
        "https://images.unsplash.com/photo-1745864383932-ce2e7edf92eb?q=80&w=1740&auto=format&fit=crop",
        use_container_width=True
    )
    st.markdown('<p class="img-caption">A busy pedestrian crossing in Shibuya, Tokyo — illustrating dense urban walkability.</p>', unsafe_allow_html=True)

    st.markdown('<hr class="soft-divider">', unsafe_allow_html=True)

    # ── Section 2: Why It Matters ─────────────────────────────────────────
    st.markdown('<p class="section-label">Significance</p>', unsafe_allow_html=True)
    st.markdown('<h2 class="section-heading">Why Walkability Matters</h2>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1], gap="large")
    with col1:
        st.markdown("""
        <p class="body-text">
        Beyond individual convenience, walkability plays a significant role in broader social,
        environmental, and economic outcomes. Communities with higher walkability often experience
        reduced traffic congestion, lower transportation costs for households, and decreased greenhouse
        gas emissions. Walkable neighborhoods can also support local businesses by increasing foot
        traffic and strengthening neighborhood economies.
        </p>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <p class="body-text">
        Public health researchers have linked walkable environments to lower rates of chronic diseases
        such as obesity and cardiovascular conditions. At the same time, walkability is not distributed
        evenly across all neighborhoods, raising important questions about spatial equity and access to
        opportunity. Historically marginalized communities may face barriers such as poor infrastructure,
        long distances to essential services, or unsafe pedestrian conditions.
        </p>
        """, unsafe_allow_html=True)

    # ── Topic tags ────────────────────────────────────────────────────────
    st.markdown("""
    <div class="tag-row">
        <span class="tag">Public Health</span>
        <span class="tag">Urban Equity</span>
        <span class="tag">Climate & Emissions</span>
        <span class="tag">Transportation</span>
        <span class="tag">Land Use</span>
        <span class="tag">Community Development</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<hr class="soft-divider">', unsafe_allow_html=True)

    # ── Section 3: The Dataset ────────────────────────────────────────────
    st.markdown('<p class="section-label">Data & Methodology</p>', unsafe_allow_html=True)
    st.markdown('<h2 class="section-heading">The EPA National Walkability Index</h2>', unsafe_allow_html=True)

    col1, col2 = st.columns([3, 2], gap="large")
    with col1:
        st.markdown("""
        <p class="body-text">
        This analysis utilizes the Walkability Index developed by the United States Environmental
        Protection Agency as part of its Smart Location Database. The index is constructed using
        standardized measures of intersection density, proximity to transit stops, employment mix,
        and employment-to-household mix. Each component is ranked nationally and weighted to produce
        a composite score ranging from 1 to 20, where higher values indicate greater walkability.
        </p>
        <br>
        <p class="body-text">
        Because the methodology applies consistent national scaling, it allows for direct comparison
        across metropolitan regions, suburban areas, and rural communities. The use of nationally
        standardized indicators ensures methodological rigor and supports reproducible geographic analysis.
        </p>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div style="background:#1e293b;border:1px solid #334155;border-radius:12px;padding:24px;color:#cbd5e1;">
            <p style="font-size:13px;font-weight:700;color:#3498db;letter-spacing:1.5px;
            text-transform:uppercase;margin-bottom:16px;">Index Components</p>
            <div style="margin-bottom:14px;">
                <div style="font-size:14px;font-weight:600;color:#f1f5f9;margin-bottom:4px;">Intersection Density</div>
                <div style="font-size:13px;color:#94a3b8;">Street network connectivity</div>
            </div>
            <div style="margin-bottom:14px;">
                <div style="font-size:14px;font-weight:600;color:#f1f5f9;margin-bottom:4px;">Transit Proximity</div>
                <div style="font-size:13px;color:#94a3b8;">Distance to nearest transit stop</div>
            </div>
            <div style="margin-bottom:14px;">
                <div style="font-size:14px;font-weight:600;color:#f1f5f9;margin-bottom:4px;">Employment Mix</div>
                <div style="font-size:13px;color:#94a3b8;">Diversity of nearby job types</div>
            </div>
            <div>
                <div style="font-size:14px;font-weight:600;color:#f1f5f9;margin-bottom:4px;">Employment-Household Mix</div>
                <div style="font-size:13px;color:#94a3b8;">Balance of jobs and residents</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Image 2 ───────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.image(
        "https://images.unsplash.com/photo-1494526585095-c41746248156",
        use_container_width=True
    )
    st.markdown('<p class="img-caption">Dense urban environments typically exhibit higher intersection density and mixed land use patterns.</p>', unsafe_allow_html=True)

    st.markdown('<hr class="soft-divider">', unsafe_allow_html=True)

    # ── Section 4: Data Science Angle ─────────────────────────────────────
    st.markdown('<p class="section-label">Analytical Approach</p>', unsafe_allow_html=True)
    st.markdown('<h2 class="section-heading">From Mapping to Machine Learning</h2>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1], gap="large")
    with col1:
        st.markdown("""
        <p class="body-text">
        Urban planning and transportation policies increasingly emphasize walkability as a tool for
        improving quality of life and addressing climate challenges. Measuring walkability allows
        policymakers and planners to identify areas of need and opportunity. Understanding patterns
        of walkability can inform decisions related to zoning, transportation investment, and community
        development.
        </p>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <p class="body-text">
        The growing availability of standardized, large-scale urban datasets has opened new
        opportunities to analyze walkability predictively. By applying machine learning techniques
        to built-environment indicators, researchers can identify which combinations of neighborhood
        characteristics most reliably distinguish highly walkable areas from car-dependent ones —
        enabling cities to anticipate the walkability implications of proposed developments before
        construction begins.
        </p>
        """, unsafe_allow_html=True)

    st.markdown('<hr class="soft-divider">', unsafe_allow_html=True)

    # ── Research Questions ────────────────────────────────────────────────
    st.markdown('<p class="section-label">Research Questions</p>', unsafe_allow_html=True)
    st.markdown('<h2 class="section-heading">What This Analysis Explores</h2>', unsafe_allow_html=True)

    questions = [
        "How does walkability vary across neighborhoods in the United States?",
        "What characteristics distinguish highly walkable areas from less walkable ones?",
        "Are walkable neighborhoods concentrated in specific regions or urban forms?",
        "How evenly is walkability distributed across communities?",
        "What role does street connectivity play in overall walkability?",
        "How does access to destinations influence walkability outcomes?",
        "Are there identifiable spatial patterns in walkability levels?",
        "How might walkability reflect broader issues of urban accessibility?",
        "Do certain neighborhood characteristics consistently align with higher walkability?",
        "How can walkability insights support more livable communities?"
    ]

    col1, col2 = st.columns(2, gap="large")
    for i, q in enumerate(questions, start=1):
        target = col1 if i <= 5 else col2
        with target:
            st.markdown(f"""
            <div class="q-card">
                <span class="q-num">{i:02d}</span>
                <span class="q-text">{q}</span>
            </div>
            """, unsafe_allow_html=True)

    # ── Lottie animation ──────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div style="display:flex;justify-content:center;">', unsafe_allow_html=True)
    components.html(
        """
        <iframe src="https://lottie.host/embed/ba69d7f2-9f31-4ba0-a6d1-bdbdc8981805/BeN9Sdo2gO.lottie"
                style="width:100%;max-width:500px;height:120px;border:none;"></iframe>
        """,
        height=120
    )
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('<div style="margin-bottom:60px;"></div>', unsafe_allow_html=True)