import streamlit as st
import pandas as pd
import os
from pathlib import Path

def app():
    
    st.markdown("""
    <style>
        .section-header {
            font-size: 28px;
            font-weight: 600;
            color: #2c3e50;
            margin-top: 40px;
            margin-bottom: 20px;
            border-bottom: 2px solid #3498db;
            padding-bottom: 10px;
        }
        
        .step-title {
            font-size: 22px;
            font-weight: 600;
            color: #3498db;
            margin-top: 30px;
            margin-bottom: 15px;
        }
        
        .step-description {
            font-size: 16px;
            line-height: 1.8;
            color: #3498db;
            margin-bottom: 20px;
        }
        
        .metric-box {
            background-color: #f8f9fa;
            border-left: 4px solid #3498db;
            padding: 15px 20px;
            margin: 10px 0;
            border-radius: 5px;
        }
        
        .metric-value {
            font-size: 24px;
            font-weight: 700;
            color: #2c3e50;
        }
        
        .metric-label {
            font-size: 14px;
            color: #7f8c8d;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Page title
    st.markdown('<h1 class="section-header">Data Preparation Pipeline</h1>', unsafe_allow_html=True)
    
    # Introduction
    st.markdown("""
    <p class="step-description">
    This section documents the comprehensive data cleaning and preparation process applied to the EPA National 
    Walkability Index dataset. Each step ensures data quality, consistency, and readiness for analysis. The 
    pipeline follows industry best practices for geospatial and statistical data processing.
    </p>
    """, unsafe_allow_html=True)
    
    # Overview metrics
    st.markdown('<h2 class="section-header">Dataset Overview</h2>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-box">
            <div class="metric-label">Total Records</div>
            <div class="metric-value">220,740</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-box">
            <div class="metric-label">Geographic Units</div>
            <div class="metric-value">Block Groups</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-box">
            <div class="metric-label">Coverage</div>
            <div class="metric-value">Nationwide</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-box">
            <div class="metric-label">Data Source</div>
            <div class="metric-value">EPA 2021</div>
        </div>
        """, unsafe_allow_html=True)
    
    
    # STEP 1: DATA LOADING
    st.markdown('<h2 class="step-title">Step 1: Data Loading</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <p class="step-description">
    The raw dataset was obtained from the EPA Smart Location Database v3.0 (January 2021). This comprehensive 
    dataset contains 117 variables describing transportation and land use characteristics for every Census block 
    group in the United States.
    </p>
    """, unsafe_allow_html=True)
    
    # Check if images exist
    if os.path.exists("images/step1_raw_data_info.txt"):
        with open("images/step1_raw_data_info.txt", "r") as f:
            st.code(f.read(), language="text")
    else:
        st.info("Run `clean_data.py` locally to generate data statistics")
    
    # STEP 2: COLUMN SELECTION
    st.markdown('<h2 class="step-title">Step 2: Column Selection</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <p class="step-description">
    From the 117 available variables, we selected 30 key columns essential for walkability analysis. This includes 
    the four core walkability indicators (intersection density, transit proximity, employment mix, and 
    employment-household mix) along with geographic identifiers and demographic context variables.
    </p>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("**Selected Variable Categories:**")
        st.markdown("""
        - Geographic Identifiers (10 variables)
        - Core Walkability Metrics (4 variables)
        - Ranked Walkability Scores (4 variables)
        - Population & Housing (3 variables)
        - Employment Data (3 variables)
        - Auto Ownership (6 variables)
        """)
    
    with col2:
        if os.path.exists("images/step2_column_selection.png"):
            st.image("images/step2_column_selection.png", use_column_width=True)
        else:
            st.info("Visualization will appear after running clean_data.py")
    
    # STEP 3: DUPLICATE REMOVAL
    st.markdown('<h2 class="step-title">Step 3: Duplicate Detection & Removal</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <p class="step-description">
    Each Census block group should have a unique GEOID10 identifier. We checked for duplicate records based on 
    this identifier and removed any duplicates, keeping the first occurrence to maintain data integrity.
    </p>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("**Duplicate Check Results:**")
        st.markdown("""
        - ✓ GEOID10 duplicates identified
        - ✓ Complete row duplicates detected
        - ✓ First occurrence retained
        - ✓ Geographic completeness maintained
        """)
    
    with col2:
        if os.path.exists("images/step3_duplicates.png"):
            st.image("images/step3_duplicates.png", use_column_width=True)
        else:
            st.info("Visualization will appear after running clean_data.py")
    
    # STEP 4: MISSING VALUE ANALYSIS
    st.markdown('<h2 class="step-title">Step 4: Missing Value Analysis</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <p class="step-description">
    Missing values were systematically identified and handled. The D4A variable (transit proximity) contains 
    -99999 values indicating "no transit service," which is valid data representing areas without public 
    transportation access. These values were retained as they provide important information about transit 
    availability.
    </p>
    """, unsafe_allow_html=True)
    
    if os.path.exists("images/step4_missing_values_before.png"):
        st.image("images/step4_missing_values_before.png", use_column_width=True)
    else:
        st.info("Visualization will appear after running clean_data.py")
    
    # STEP 5: DATA VALIDATION
    st.markdown('<h2 class="step-title">Step 5: Data Range Validation & Outlier Detection</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <p class="step-description">
    All variables were validated against expected ranges defined in the EPA methodology. Ranked scores should 
    range from 1-20, and the National Walkability Index should also fall within this range. Outliers were 
    identified but retained to preserve the full distribution of walkability conditions across the nation.
    </p>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("**Validation Criteria:**")
        st.markdown("""
        - D2A_Ranked: 1 to 20
        - D2B_Ranked: 1 to 20
        - D3B_Ranked: 1 to 20
        - D4A_Ranked: 1 to 20
        - NatWalkInd: 1 to 20
        """)
        st.success("✓ All variables within expected ranges")
    
    with col2:
        if os.path.exists("images/step5_distributions.png"):
            st.image("images/step5_distributions.png", use_column_width=True)
        else:
            st.info("Visualization will appear after running clean_data.py")
    
    # STEP 6: CALCULATION VERIFICATION
    st.markdown('<h2 class="step-title">Step 6: Walkability Index Calculation Verification</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <p class="step-description">
    The National Walkability Index is calculated using a weighted formula combining the four ranked variables. 
    We verified the calculation by recomputing the index and comparing it to the provided values.
    </p>
    """, unsafe_allow_html=True)
    
    # Display formula
    st.markdown("**Formula:**")
    st.latex(r'''
    NatWalkInd = \frac{w}{3} + \frac{x}{3} + \frac{y}{6} + \frac{z}{6}
    ''')
    
    st.markdown("""
    Where:
    - w = D3B_Ranked (Intersection Density)
    - x = D4A_Ranked (Transit Proximity)
    - y = D2B_Ranked (Employment Mix)
    - z = D2A_Ranked (Employment-Household Mix)
    """)
    
    if os.path.exists("images/step6_calculation_verification.png"):
        st.image("images/step6_calculation_verification.png", use_column_width=True)
    else:
        st.info("Visualization will appear after running clean_data.py")
    
    # STEP 7: CATEGORIZATION
    st.markdown('<h2 class="step-title">Step 7: Walkability Level Categorization</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <p class="step-description">
    Block groups were categorized into four walkability levels based on their National Walkability Index scores, 
    following the EPA methodology guidelines. This categorization enables easier interpretation and comparison 
    of walkability across different areas.
    </p>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("**Walkability Categories:**")
        st.markdown("""
        - **Least Walkable:** 1.00 - 5.75
        - **Below Average:** 5.76 - 10.50
        - **Above Average:** 10.51 - 15.25
        - **Most Walkable:** 15.26 - 20.00
        """)
    
    with col2:
        if os.path.exists("images/step7_categories.png"):
            st.image("images/step7_categories.png", use_column_width=True)
        else:
            st.info("Visualization will appear after running clean_data.py")
    
    # STEP 8: SUMMARY STATISTICS
    st.markdown('<h2 class="step-title">Step 8: Summary Statistics Generation</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <p class="step-description">
    Comprehensive statistical summaries were generated for all key variables to understand their distributions, 
    central tendencies, and variability. These statistics provide baseline metrics for subsequent analysis.
    </p>
    """, unsafe_allow_html=True)
    
    if os.path.exists("images/step8_summary_statistics.csv"):
        summary_df = pd.read_csv("images/step8_summary_statistics.csv", index_col=0)
        st.dataframe(summary_df, use_container_width=True)
    else:
        st.info("Summary statistics will appear after running clean_data.py")
    
    if os.path.exists("images/step8_summary_boxplots.png"):
        st.image("images/step8_summary_boxplots.png", use_column_width=True)
    
    # STEP 9: DATA EXPORT
    
    st.markdown('<h2 class="step-title">Step 9: Cleaned Data Export</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <p class="step-description">
    The cleaned dataset was exported in multiple formats (CSV and Excel) to ensure compatibility with various 
    analysis tools. Metadata documenting the cleaning process was also saved for reproducibility and transparency.
    </p>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Output Files:**")
        st.markdown("""
        - `walkability_cleaned.csv`
        - `walkability_cleaned.xlsx`
        - `cleaning_metadata.txt`
        """)
    
    with col2:
        st.markdown("**File Locations:**")
        st.markdown("""
        - Data: `cleaned_data/`
        - Images: `images/`
        - Metadata: `cleaned_data/`
        """)
    
    # STEP 10: BEFORE/AFTER COMPARISON
    st.markdown('<h2 class="step-title">Step 10: Before & After Comparison</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <p class="step-description">
    A comprehensive comparison of the dataset before and after cleaning demonstrates the impact of each 
    processing step. This comparison ensures transparency and validates the cleaning pipeline effectiveness.
    </p>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        if os.path.exists("images/step10_before_after_comparison.csv"):
            comparison_df = pd.read_csv("images/step10_before_after_comparison.csv")
            st.dataframe(comparison_df, use_container_width=True, hide_index=True)
        else:
            st.info("Comparison table will appear after running clean_data.py")
    
    with col2:
        if os.path.exists("images/step10_before_after.png"):
            st.image("images/step10_before_after.png", use_column_width=True)
        else:
            st.info("Visualization will appear after running clean_data.py")
    
    # FINAL SUMMARY
    st.markdown('<h2 class="section-header">Cleaning Pipeline Summary</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <p class="step-description">
    The data preparation pipeline successfully processed the EPA National Walkability Index dataset through 
    10 systematic steps, ensuring data quality, consistency, and analytical readiness. All geographic units 
    were retained to maintain complete nationwide coverage.
    </p>
    """, unsafe_allow_html=True)
    
    # Summary metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-box">
            <div class="metric-label">Data Quality</div>
            <div class="metric-value">✓ Verified</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-box">
            <div class="metric-label">Calculations</div>
            <div class="metric-value">✓ Validated</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-box">
            <div class="metric-label">Geographic Coverage</div>
            <div class="metric-value">✓ Complete</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Next steps callout
    st.markdown("<div style='margin-top: 50px;'>", unsafe_allow_html=True)
    st.success("✓ Data preparation complete. Proceed to the Models section to explore walkability patterns and analysis.")
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Spacer
    st.markdown('<div style="margin-bottom: 60px;"></div>', unsafe_allow_html=True)