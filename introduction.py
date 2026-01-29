import streamlit as st
import streamlit.components.v1 as components

def app():
    
    st.markdown("""
    <style>
        /* Hide Streamlit branding */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        
        /* Professional typography */
        .main-title {
            font-size: 85px;
            font-weight: 700;
            color: #1a1a1a;
            text-align: center;
            margin-bottom: 8px;
            letter-spacing: -0.5px;
        }
        
        .subtitle {
            font-size: 18px;
            color: #666666;
            text-align: center;
            margin-bottom: 50px;
            font-weight: 400;
        }
        
        .section-header {
            font-size: 28px;
            font-weight: 600;
            color: #2c3e50;
            margin-top: 60px;
            margin-bottom: 30px;
            border-bottom: 2px solid #e0e0e0;
            padding-bottom: 10px;
        }
        
        .body-text {
            font-size: 17px;
            line-height: 2.1;
            color: #F0F8FF;
            text-align: justify;
            margin-bottom: 35px;
        }
        
        .image-caption {
            text-align: center;
            font-size: 14px;
            color: #777777;
            font-style: italic;
            margin-top: 12px;
            margin-bottom: 40px;
        }
        
        .question-item {
            font-size: 16px;
            line-height: 1.9;
            color: #3498db;
            margin-bottom: 18px;
            padding-left: 8px;
            border-left: 3px solid #3498db;
            padding: 12px 0 12px 20px;
        }
        
        .question-number {
            font-weight: 600;
            color: #3498db;
            margin-right: 8px;
        }
        
        /* Animation container */
        .lottie-container {
            display: flex;
            justify-content: center;
            margin: 40px 0;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Header section
    st.markdown('<h1 class="main-title">Walkability Analysis</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Understanding How Urban Design Shapes Movement in American Cities</p>', unsafe_allow_html=True)
    
    # Introduction section
    st.markdown('<h2 class="section-header">Introduction</h2>', unsafe_allow_html=True)
    
    # Paragraph 1 
    st.markdown("""
    <p class="body-text">
    Walkability refers to how easily people can move through their neighborhoods on foot to reach everyday 
    destinations such as grocery stores, schools, workplaces, parks, and healthcare services. In walkable 
    environments, streets are designed to support pedestrians through connected road networks, safe crossings, 
    and a mix of land uses that place essential services within reasonable walking distances.
    </p>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <p class="body-text">
    These environments encourage daily physical activity, reduce dependence on automobiles, and promote stronger 
    social interactions among residents. Walkability has become an increasingly important concept as cities grow 
    denser and communities seek more sustainable forms of mobility. Neighborhoods that support walking tend to 
    foster healthier lifestyles, improved air quality, and greater economic vitality.
    </p>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <p class="body-text">
    The presence of sidewalks, intersections, and nearby destinations shapes how people experience their 
    surroundings and interact with their community. Walkability also influences how accessible a city feels to 
    individuals who do not drive, including children, older adults, and people with disabilities. As urban 
    populations continue to expand, understanding walkability has become essential for designing livable and 
    inclusive spaces. Cities across the United States vary widely in how they prioritize pedestrian movement. 
    Examining walkability provides insight into how urban design decisions affect daily life.
    </p>
    """, unsafe_allow_html=True)

    # Paragraph 2
    st.markdown("""
    <p class="body-text">
    Beyond individual convenience, walkability plays a significant role in broader social, environmental, and 
    economic outcomes. Communities with higher walkability often experience reduced traffic congestion, lower 
    transportation costs for households, and decreased greenhouse gas emissions. Walkable neighborhoods can also 
    support local businesses by increasing foot traffic and strengthening neighborhood economies.
    </p>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <p class="body-text">
    Public health researchers have linked walkable environments to lower rates of chronic diseases such as obesity 
    and cardiovascular conditions. At the same time, walkability is not distributed evenly across all neighborhoods, 
    raising important questions about spatial equity and access to opportunity. Historically marginalized communities 
    may face barriers such as poor infrastructure, long distances to essential services, or unsafe pedestrian 
    conditions.
    </p>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <p class="body-text">
    Urban planning and transportation policies increasingly emphasize walkability as a tool for improving quality 
    of life and addressing climate challenges. Measuring walkability allows policymakers and planners to identify 
    areas of need and opportunity. Understanding patterns of walkability can inform decisions related to zoning, 
    transportation investment, and community development. As cities adapt to changing environmental and social 
    pressures, walkability remains a central component of sustainable urban living.
    </p>
    """, unsafe_allow_html=True)
    
    # Image 
    st.markdown('<div style="margin: 50px 0 20px 0;">', unsafe_allow_html=True)
    st.image("walk_image.jpg", use_container_width=True)
    
    
    st.markdown("""
    <p class="image-caption">
    Walkable neighborhoods emphasize pedestrian connectivity, mixed land use, and human-scale urban design.
    </p>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Research Questions section
    st.markdown('<h2 class="section-header">Research Questions</h2>', unsafe_allow_html=True)
    
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
    
    # Display questions in two columns
    col1, col2 = st.columns(2)
    
    for i, question in enumerate(questions, start=1):
        if i <= 5:
            with col1:
                st.markdown(
                    f'<div class="question-item"><span class="question-number">{i}.</span>{question}</div>',
                    unsafe_allow_html=True
                )
        else:
            with col2:
                st.markdown(
                    f'<div class="question-item"><span class="question-number">{i}.</span>{question}</div>',
                    unsafe_allow_html=True
                )
    
    # Animation at bottom
    st.markdown('<div class="lottie-container">', unsafe_allow_html=True)
    components.html(
        """
        <iframe src="https://lottie.host/embed/ba69d7f2-9f31-4ba0-a6d1-bdbdc8981805/BeN9Sdo2gO.lottie" 
                style="width: 100%; max-width: 500px; height: 120px; border: none;">
        </iframe>
        """,
        height=120
    )
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Spacer at bottom
    st.markdown('<div style="margin-bottom: 60px;"></div>', unsafe_allow_html=True)
