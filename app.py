"""
🎓 Student Disengagement Prediction System
Beautiful, Modern UI with Advanced Analytics
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
from datetime import datetime
from pathlib import Path
import sys

# Add project root
sys.path.append(str(Path(__file__).parent))

from config import *

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="Student Analytics AI",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "Student Disengagement Prediction System - AI-Powered Analytics"
    }
)

# ==================== CUSTOM CSS ====================
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    
    /* Hide Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {display: none;}
    
    /* Main container - Dark theme with gradient */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #334155 100%);
    }
    
    .main {
        background: transparent;
        font-family: 'Inter', sans-serif;
    }
    
    /* Remove white boxes */
    .element-container {
        background: transparent !important;
    }
    
    /* Sidebar styling - Dark sleek design */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%) !important;
        border-right: 2px solid rgba(139, 92, 246, 0.3);
    }
    
    section[data-testid="stSidebar"] > div {
        background: transparent !important;
    }
    
    section[data-testid="stSidebar"] .stMarkdown {
        color: #e2e8f0 !important;
    }
    
    section[data-testid="stSidebar"] h1, 
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: #8b5cf6 !important;
        font-weight: 700;
    }
    
    /* Radio buttons in sidebar */
    section[data-testid="stSidebar"] .stRadio > label {
        color: #e2e8f0 !important;
        font-weight: 600;
        font-size: 1.1rem;
    }
    
    section[data-testid="stSidebar"] [role="radiogroup"] label {
        background: rgba(139, 92, 246, 0.1);
        padding: 0.75rem 1rem;
        border-radius: 8px;
        margin: 0.3rem 0;
        transition: all 0.3s ease;
        border: 1px solid rgba(139, 92, 246, 0.2);
    }
    
    section[data-testid="stSidebar"] [role="radiogroup"] label:hover {
        background: rgba(139, 92, 246, 0.2);
        border-color: rgba(139, 92, 246, 0.5);
        transform: translateX(5px);
    }
    
    /* Card styling - Glassmorphism */
    .metric-card {
        background: rgba(30, 41, 59, 0.7) !important;
        backdrop-filter: blur(16px) saturate(180%);
        padding: 1.5rem !important;
        border-radius: 20px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3), 
                    inset 0 1px 0 rgba(255, 255, 255, 0.1);
        border: 1px solid rgba(139, 92, 246, 0.2);
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        margin: 1rem 0;
        position: relative;
        z-index: 1;
        overflow: visible;
        display: block;
        min-height: 120px;
    }
    
    .metric-card:hover {
        transform: translateY(-8px) scale(1.02);
        box-shadow: 0 16px 48px rgba(139, 92, 246, 0.3);
        border-color: rgba(139, 92, 246, 0.5);
        z-index: 2;
    }
    
    /* Fix metric containers inside cards */
    .metric-card > div {
        background: transparent !important;
    }
    
    .metric-card [data-testid="stMetric"] {
        background: transparent !important;
        padding: 0 !important;
    }
    
    /* Gradient text - Vibrant */
    .gradient-text {
        background: linear-gradient(135deg, #8b5cf6 0%, #ec4899 50%, #f59e0b 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: 900;
        font-size: 3.5rem;
        text-align: center;
        margin: 2rem 0;
        letter-spacing: -0.02em;
        animation: gradient-shift 3s ease infinite;
    }
    
    @keyframes gradient-shift {
        0%, 100% { filter: hue-rotate(0deg); }
        50% { filter: hue-rotate(10deg); }
    }
    
    /* Metric values */
    [data-testid="stMetricValue"] {
        font-size: 2.5rem !important;
        font-weight: 800 !important;
        color: #8b5cf6 !important;
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 0.9rem !important;
        font-weight: 600 !important;
        color: #94a3b8 !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    [data-testid="stMetricDelta"] {
        font-size: 0.9rem !important;
        font-weight: 600 !important;
    }
    
    /* Remove all white backgrounds */
    [data-testid="stVerticalBlock"] {
        background: transparent !important;
    }
    
    [data-testid="stHorizontalBlock"] {
        background: transparent !important;
    }
    
    [data-testid="column"] {
        background: transparent !important;
    }
    
    div.block-container {
        background: transparent !important;
        padding-top: 2rem;
    }
    
    /* Button styling - Neon glow */
    .stButton>button {
        background: linear-gradient(135deg, #8b5cf6 0%, #ec4899 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-weight: 700;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 20px rgba(139, 92, 246, 0.4);
    }
    
    .stButton>button:hover {
        transform: translateY(-3px) scale(1.05);
        box-shadow: 0 8px 30px rgba(236, 72, 153, 0.6);
    }
    
    /* Input fields - Dark theme */
    .stTextInput>div>div>input,
    .stSelectbox>div>div>div,
    .stNumberInput>div>div>input,
    .stSlider>div>div>div {
        background: rgba(30, 41, 59, 0.8) !important;
        border: 2px solid rgba(139, 92, 246, 0.3) !important;
        border-radius: 10px !important;
        color: #e2e8f0 !important;
        padding: 0.75rem !important;
        font-size: 1rem;
        font-weight: 500;
    }
    
    .stTextInput>div>div>input:focus,
    .stSelectbox>div>div>div:focus {
        border-color: #8b5cf6 !important;
        box-shadow: 0 0 0 3px rgba(139, 92, 246, 0.2) !important;
        outline: none !important;
    }
    
    /* Selectbox dropdown */
    [data-baseweb="select"] {
        background: rgba(30, 41, 59, 0.8) !important;
    }
    
    [data-baseweb="select"] > div {
        background: rgba(30, 41, 59, 0.8) !important;
        border-color: rgba(139, 92, 246, 0.3) !important;
        color: #e2e8f0 !important;
    }
    
    /* Dropdown menu */
    [role="listbox"] {
        background: rgba(30, 41, 59, 0.95) !important;
        border: 1px solid rgba(139, 92, 246, 0.3) !important;
    }
    
    [role="option"] {
        background: transparent !important;
        color: #e2e8f0 !important;
    }
    
    [role="option"]:hover {
        background: rgba(139, 92, 246, 0.2) !important;
    }
    
    /* Labels */
    label {
        color: #e2e8f0 !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        margin-bottom: 0.5rem !important;
    }
    
    /* Risk badges - Animated */
    .risk-high {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
        color: white;
        padding: 0.6rem 1.2rem;
        border-radius: 25px;
        font-weight: 800;
        display: inline-block;
        box-shadow: 0 4px 15px rgba(239, 68, 68, 0.4);
        animation: pulse-red 2s ease-in-out infinite;
    }
    
    @keyframes pulse-red {
        0%, 100% { box-shadow: 0 4px 15px rgba(239, 68, 68, 0.4); }
        50% { box-shadow: 0 4px 25px rgba(239, 68, 68, 0.7); }
    }
    
    .risk-medium {
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
        color: white;
        padding: 0.6rem 1.2rem;
        border-radius: 25px;
        font-weight: 800;
        display: inline-block;
        box-shadow: 0 4px 15px rgba(245, 158, 11, 0.4);
        animation: pulse-orange 2s ease-in-out infinite;
    }
    
    @keyframes pulse-orange {
        0%, 100% { box-shadow: 0 4px 15px rgba(245, 158, 11, 0.4); }
        50% { box-shadow: 0 4px 25px rgba(245, 158, 11, 0.7); }
    }
    
    .risk-low {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        padding: 0.6rem 1.2rem;
        border-radius: 25px;
        font-weight: 800;
        display: inline-block;
        box-shadow: 0 4px 15px rgba(16, 185, 129, 0.4);
    }
    
    /* Info cards - Glass design */
    .info-card {
        background: rgba(139, 92, 246, 0.15) !important;
        backdrop-filter: blur(10px);
        padding: 1.5rem;
        border-radius: 16px;
        border: 1px solid rgba(139, 92, 246, 0.3);
        margin: 1rem 0;
        box-shadow: 0 4px 20px rgba(0,0,0,0.2);
        color: #e2e8f0 !important;
        font-weight: 600;
    }
    
    /* DataFrames - Dark theme */
    [data-testid="stDataFrame"] {
        background: rgba(30, 41, 59, 0.7) !important;
        border-radius: 12px;
        border: 1px solid rgba(139, 92, 246, 0.2);
    }
    
    /* Plotly charts - Remove white bg */
    .js-plotly-plot {
        background: transparent !important;
    }
    
    /* Animations */
    @keyframes fadeIn {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .fade-in {
        animation: fadeIn 0.6s ease-out;
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(15, 23, 42, 0.5);
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #8b5cf6 0%, #ec4899 100%);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #a78bfa 0%, #f472b6 100%);
    }
</style>
""", unsafe_allow_html=True)

# ==================== HELPER FUNCTIONS ====================

@st.cache_resource
def load_model_and_data():
    """Load ML model and data"""
    try:
        model = joblib.load(MODEL_FILE)
        df = pd.read_csv(CLEANED_DATA_FILE)
        scaler = joblib.load(SCALER_FILE)
        feature_names = joblib.load(FEATURE_NAMES_FILE)
        return model, df, scaler, feature_names
    except:
        return None, None, None, None

def get_risk_category(score):
    """Get risk category and styling"""
    if score >= RISK_THRESHOLDS['medium']:
        return "High Risk", "risk-high", "🔴", "#ef4444"
    elif score >= RISK_THRESHOLDS['low']:
        return "Medium Risk", "risk-medium", "🟡", "#f59e0b"
    else:
        return "Low Risk", "risk-low", "🟢", "#10b981"

def create_gauge_chart(value, title="Risk Score"):
    """Create beautiful gauge chart"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 24, 'color': '#e2e8f0', 'family': 'Arial Black'}},
        number={'suffix': "%", 'font': {'size': 40, 'color': '#e2e8f0'}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 2, 'tickcolor': "#cbd5e1"},
            'bar': {'color': "#8b5cf6", 'thickness': 0.75},
            'bgcolor': "rgba(30, 41, 59, 0.5)",
            'borderwidth': 2,
            'bordercolor': "rgba(139, 92, 246, 0.3)",
            'steps': [
                {'range': [0, 30], 'color': 'rgba(16, 185, 129, 0.3)'},
                {'range': [30, 60], 'color': 'rgba(245, 158, 11, 0.3)'},
                {'range': [60, 100], 'color': 'rgba(239, 68, 68, 0.3)'}
            ],
            'threshold': {
                'line': {'color': "#ef4444", 'width': 4},
                'thickness': 0.75,
                'value': 60
            }
        }
    ))
    
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={'family': "Arial", 'color': '#e2e8f0'},
        height=300,
        margin=dict(l=20, r=20, t=60, b=20)
    )
    return fig

def process_uploaded_data(uploaded_file, model, scaler, features):
    """Process uploaded Excel or CSV file and make predictions"""
    try:
        # Read the uploaded file
        if uploaded_file.name.endswith('.csv'):
            df_new = pd.read_csv(uploaded_file)
        else:
            df_new = pd.read_excel(uploaded_file)
        
        # Store original data
        df_original = df_new.copy()
        
        # Expected columns from raw data (dropout is not required - we will predict it)
        expected_cols = ['student_id', 'gender', 'department', 'scholarship', 'parental_education',
                        'extra_curricular', 'age', 'cgpa', 'attendance_rate', 'family_income',
                        'past_failures', 'study_hours_per_week', 'assignments_submitted', 
                        'projects_completed', 'total_activities', 'sports_participation']
        
        missing_cols = [col for col in expected_cols if col not in df_new.columns]
        if missing_cols:
            return None, f"Missing required columns: {', '.join(missing_cols)}", None
        
        # Ensure all columns are the correct type
        df_new['student_id'] = df_new['student_id'].astype(str)
        df_new['gender'] = df_new['gender'].astype(str)
        df_new['department'] = df_new['department'].astype(str)
        df_new['scholarship'] = df_new['scholarship'].astype(str)
        df_new['parental_education'] = df_new['parental_education'].astype(str)
        df_new['sports_participation'] = df_new['sports_participation'].astype(str)
        
        # Ensure numeric columns
        numeric_cols = ['extra_curricular', 'age', 'cgpa', 'attendance_rate', 'family_income',
                       'past_failures', 'study_hours_per_week', 'assignments_submitted', 
                       'projects_completed', 'total_activities']
        for col in numeric_cols:
            df_new[col] = pd.to_numeric(df_new[col], errors='coerce')
        
        # ===== EXACT PREPROCESSING FROM preprocess.py =====
        
        # Standardize gender (EXACT same logic)
        df_new['gender'] = df_new['gender'].str.upper().replace({'M': 'Male', 'FEMALE': 'Female', 'NA': 'Other', 'NAN': 'Other'})
        df_new['gender'] = df_new['gender'].fillna('Other')
        
        # Standardize department
        df_new['department'] = df_new['department'].str.upper()
        df_new['department'] = df_new['department'].fillna('UNKNOWN')
        
        # Standardize scholarship (EXACT same logic)
        df_new['scholarship'] = df_new['scholarship'].str.upper().replace({'Y': 'Yes', 'N': 'No', 'NOPE': 'No', 'NAN': 'No'})
        df_new['scholarship'] = df_new['scholarship'].fillna('No')
        df_new['scholarship_binary'] = (df_new['scholarship'] == 'Yes').astype(int)
        
        # Standardize sports participation (EXACT same logic)
        df_new['sports_participation'] = df_new['sports_participation'].str.upper().replace({'Y': 'Yes', 'N': 'No', 'NAN': 'No'})
        df_new['sports_participation'] = df_new['sports_participation'].fillna('No')
        df_new['sports_binary'] = (df_new['sports_participation'] == 'Yes').astype(int)
        
        # Handle parental education
        df_new['parental_education'] = df_new['parental_education'].fillna('Unknown')
        df_new['parental_education'] = df_new['parental_education'].replace({'nan': 'Unknown', 'NA': 'Unknown', 'NAN': 'Unknown'})
        
        # Fill numeric missing values with median (or 0 if all values are NaN)
        for col in numeric_cols:
            median_val = df_new[col].median()
            df_new[col] = df_new[col].fillna(median_val if not pd.isna(median_val) else 0)
        
        # Handle outliers in family_income (negative values)
        df_new['family_income'] = df_new['family_income'].abs()
        
        # ===== FEATURE ENGINEERING - MATCH train_model.py EXACTLY =====
        
        # Risk flags
        df_new['low_attendance'] = (df_new['attendance_rate'] < 70).astype(int)
        df_new['low_cgpa'] = (df_new['cgpa'] < 5.0).astype(int)
        df_new['high_failures'] = (df_new['past_failures'] > 3).astype(int)
        df_new['low_study_hours'] = (df_new['study_hours_per_week'] < 10).astype(int)
        
        # Interaction features
        df_new['cgpa_attendance_interaction'] = df_new['cgpa'] * df_new['attendance_rate']
        df_new['study_cgpa_ratio'] = df_new['study_hours_per_week'] / (df_new['cgpa'] + 0.1)
        df_new['assignment_completion_rate'] = df_new['assignments_submitted'] / 50.0
        df_new['academic_performance_score'] = (df_new['cgpa'] * 10 + df_new['attendance_rate']) / 2
        
        # Engagement score (different from preprocess.py!)
        df_new['engagement_score'] = (
            df_new['sports_binary'] +
            (df_new['extra_curricular'] / 5.0)
        )
        
        # Risk score composite
        df_new['risk_score'] = (
            df_new['low_attendance'] * 3 +
            df_new['low_cgpa'] * 3 +
            df_new['high_failures'] * 2 +
            df_new['low_study_hours'] * 1
        )
        
        # Polynomial features
        df_new['cgpa_squared'] = df_new['cgpa'] ** 2
        df_new['attendance_squared'] = df_new['attendance_rate'] ** 2
        
        # Categorical binning
        df_new['income_bracket'] = pd.cut(df_new['family_income'], 
                                          bins=[0, 25000, 50000, 75000, 1000000],
                                          labels=['Low', 'Medium', 'High', 'Very High'])
        df_new['age_group'] = pd.cut(df_new['age'], 
                                      bins=[15, 18, 21, 25, 100],
                                      labels=['Teen', 'Young Adult', 'Adult', 'Mature'])
        
        # Handle NaN values from pd.cut (values outside bins)
        df_new['income_bracket'] = df_new['income_bracket'].fillna('Medium')
        df_new['age_group'] = df_new['age_group'].fillna('Young Adult')
        
        # Convert to string to ensure consistency with label encoders
        df_new['income_bracket'] = df_new['income_bracket'].astype(str)
        df_new['age_group'] = df_new['age_group'].astype(str)
        
        # Use the SAVED label encoders from training (critical for consistency)
        # Encode ALL 8 categorical columns IN-PLACE to match train_model.py
        try:
            label_encoders = joblib.load(MODEL_DIR / 'label_encoders.pkl')
            
            # List of all categorical columns that need encoding (from train_model.py)
            categorical_cols = ['gender', 'department', 'scholarship', 'parental_education',
                               'extra_curricular', 'sports_participation', 'income_bracket', 'age_group']
            
            for col in categorical_cols:
                if col in df_new.columns and col in label_encoders:
                    # Convert to string to handle any type mismatches
                    df_new[col] = df_new[col].astype(str)
                    
                    # Get known classes from the saved encoder
                    known_classes = set(label_encoders[col].classes_)
                    
                    # Map unseen values to the first known class (fallback)
                    fallback_value = label_encoders[col].classes_[0]
                    df_new[col] = df_new[col].apply(lambda x: x if x in known_classes else fallback_value)
                    
                    # Transform in-place (no _encoded suffix)
                    df_new[col] = label_encoders[col].transform(df_new[col])
        except Exception as enc_error:
            # Fallback: fit new encoders if saved ones aren't available (NOT recommended)
            st.warning(f"⚠️ Could not load saved encoders: {enc_error}. Using fallback encoding (predictions may be less accurate).")
            from sklearn.preprocessing import LabelEncoder
            
            categorical_cols = ['gender', 'department', 'scholarship', 'parental_education',
                               'extra_curricular', 'sports_participation', 'income_bracket', 'age_group']
            
            for col in categorical_cols:
                if col in df_new.columns:
                    le = LabelEncoder()
                    df_new[col] = df_new[col].astype(str)
                    df_new[col] = le.fit_transform(df_new[col])
        
        # Prepare features for prediction (use the same feature columns as training)
        X_new = df_new[features].copy()
        
        # CRITICAL: Fill any remaining NaN values before scaling
        # This handles edge cases from feature engineering
        X_new = X_new.fillna(X_new.median())
        
        # Scale features
        X_new_scaled = scaler.transform(X_new)
        
        # Make predictions
        predictions = model.predict(X_new_scaled)
        prediction_proba = model.predict_proba(X_new_scaled)[:, 1]
        
        # Add predictions (store under 'dropout' to match training target)
        df_new['dropout'] = predictions
        df_new['risk_probability'] = prediction_proba
        
        # Calculate statistics for verification
        stats = {
            'total_students': len(df_new),
            'at_risk': int(predictions.sum()),
            'safe': int(len(predictions) - predictions.sum()),
            'avg_risk_probability': float(prediction_proba.mean()),
            'high_risk_count': int((prediction_proba >= 0.6).sum()),
            'medium_risk_count': int(((prediction_proba >= 0.3) & (prediction_proba < 0.6)).sum()),
            'low_risk_count': int((prediction_proba < 0.3).sum())
        }
        
        return df_new, None, stats
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        return None, str(e) + "\n\nDetails:\n" + error_detail, None

# ==================== SIDEBAR ====================

# Initialize session state for uploaded data
if 'uploaded_df' not in st.session_state:
    st.session_state.uploaded_df = None
if 'data_source' not in st.session_state:
    st.session_state.data_source = 'default'

with st.sidebar:
    st.markdown("""
        <div style='text-align: center; padding: 1rem 0;'>
            <h1 style='color: white; font-size: 1.8rem; margin: 0;'>🎓</h1>
            <h2 style='color: white; font-size: 1.3rem; margin: 0.5rem 0;'>Student Analytics</h2>
            <p style='color: #94a3b8; font-size: 0.9rem;'>AI-Powered Insights</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Navigation with custom styling
    page = st.radio(
        "Navigation",
        ["🏠 Dashboard", "👥 Students", "🎯 Prediction", "📊 Analytics", "💬 AI Assistant"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    
    # System status
    st.markdown("""
        <div class='glass-card'>
            <h3 style='color: white; margin-top: 0;'>System Status</h3>
    """, unsafe_allow_html=True)
    
    model, df_default, scaler, features = load_model_and_data()
    
    # Use uploaded data if available, otherwise use default
    if st.session_state.data_source == 'uploaded' and st.session_state.uploaded_df is not None:
        df = st.session_state.uploaded_df
        data_label = "📤 Uploaded Data"
    else:
        df = df_default
        data_label = "📊 Default Data"
    
    if model is not None:
        st.success("✓ Model Active")
    else:
        st.error("✗ Model Not Found")
    
    if df is not None:
        st.success(f"✓ {len(df)} Students Loaded")
        st.info(data_label)
    else:
        st.warning("! Run setup first")
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Quick stats
    if df is not None:
        st.markdown("""
            <div class='glass-card'>
                <h4 style='color: white; margin-top: 0;'>Quick Stats</h4>
        """, unsafe_allow_html=True)
        
        # Ensure consistent target column name 'dropout' (preferred). If older 'disengaged' exists, map it.
        if 'dropout' not in df.columns:
            if 'disengaged' in df.columns:
                df['dropout'] = df['disengaged']
            else:
                df['dropout'] = 0
        at_risk = int(df['dropout'].sum())
        st.metric("At Risk Today", f"{at_risk}", f"{(at_risk/len(df)*100):.1f}%")
        
        st.markdown("</div>", unsafe_allow_html=True)

# ==================== MAIN CONTENT ====================

# Dashboard Page
if page == "🏠 Dashboard":
    # Header with animation
    st.markdown('<h1 class="gradient-text fade-in">Student Analytics Dashboard</h1>', unsafe_allow_html=True)
    
    if df is not None and model is not None:
        # KPI Cards
        col1, col2, col3, col4, col5 = st.columns(5)
        
        total = len(df)
        at_risk = int(df['dropout'].sum())
        avg_attendance = df['attendance_rate'].mean()
        avg_gpa = df['cgpa'].mean()
        
        # KPI Cards with custom HTML metrics
        with col1:
            st.markdown(f'''
                <div class="metric-card fade-in">
                    <div style="text-align: center;">
                        <p style="color: #94a3b8; font-size: 0.9rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 0.5rem;">TOTAL STUDENTS</p>
                        <h2 style="color: #8b5cf6; font-size: 2.5rem; font-weight: 800; margin: 0;">{total:,}</h2>
                    </div>
                </div>
            ''', unsafe_allow_html=True)
        
        with col2:
            st.markdown(f'''
                <div class="metric-card fade-in">
                    <div style="text-align: center;">
                        <p style="color: #94a3b8; font-size: 0.9rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 0.5rem;">AT RISK</p>
                        <h2 style="color: #8b5cf6; font-size: 2.5rem; font-weight: 800; margin: 0;">{at_risk}</h2>
                        <p style="color: #ef4444; font-size: 0.9rem; font-weight: 600; margin-top: 0.5rem;">↑ {(at_risk/total*100):.1f}%</p>
                    </div>
                </div>
            ''', unsafe_allow_html=True)
        
        with col3:
            st.markdown(f'''
                <div class="metric-card fade-in">
                    <div style="text-align: center;">
                        <p style="color: #94a3b8; font-size: 0.9rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 0.5rem;">SAFE</p>
                        <h2 style="color: #8b5cf6; font-size: 2.5rem; font-weight: 800; margin: 0;">{total-at_risk}</h2>
                        <p style="color: #10b981; font-size: 0.9rem; font-weight: 600; margin-top: 0.5rem;">↑ {((total-at_risk)/total*100):.1f}%</p>
                    </div>
                </div>
            ''', unsafe_allow_html=True)
        
        with col4:
            st.markdown(f'''
                <div class="metric-card fade-in">
                    <div style="text-align: center;">
                        <p style="color: #94a3b8; font-size: 0.9rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 0.5rem;">AVG ATTENDANCE</p>
                        <h2 style="color: #8b5cf6; font-size: 2.5rem; font-weight: 800; margin: 0;">{avg_attendance:.1f}%</h2>
                    </div>
                </div>
            ''', unsafe_allow_html=True)
        
        with col5:
            st.markdown(f'''
                <div class="metric-card fade-in">
                    <div style="text-align: center;">
                        <p style="color: #94a3b8; font-size: 0.9rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 0.5rem;">AVG GPA</p>
                        <h2 style="color: #8b5cf6; font-size: 2.5rem; font-weight: 800; margin: 0;">{avg_gpa:.2f}/10</h2>
                    </div>
                </div>
            ''', unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Charts Row 1
        col1, col2 = st.columns(2)
        
        with col1:
            # Risk Distribution Pie
            risk_counts = df['dropout'].value_counts()
            fig = go.Figure(data=[go.Pie(
                labels=['Safe', 'At Risk'],
                values=[risk_counts.get(0, 0), risk_counts.get(1, 0)],
                hole=0.5,
                marker=dict(colors=['#10b981', '#ef4444'], 
                           line=dict(color='rgba(30, 41, 59, 0.5)', width=3)),
                textfont=dict(size=16, color='#e2e8f0', family='Arial Black'),
                pull=[0.05, 0.1]
            )])
            fig.update_layout(
                title={'text': "Student Risk Distribution", 'x': 0.5, 'xanchor': 'center',
                      'font': {'size': 20, 'color': '#e2e8f0', 'family': 'Arial Black'}},
                showlegend=True,
                legend=dict(font=dict(size=14, color='#e2e8f0')),
                height=400,
                paper_bgcolor='rgba(30, 41, 59, 0.7)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Department Analysis
            dept_data = df.groupby('department')['dropout'].agg(['sum', 'count']).reset_index()
            dept_data['safe'] = dept_data['count'] - dept_data['sum']
            
            fig = go.Figure()
            fig.add_trace(go.Bar(name='At Risk', x=dept_data['department'], y=dept_data['sum'],
                                marker_color='#ef4444', text=dept_data['sum'], textposition='auto'))
            fig.add_trace(go.Bar(name='Safe', x=dept_data['department'], y=dept_data['safe'],
                                marker_color='#10b981', text=dept_data['safe'], textposition='auto'))
            
            fig.update_layout(
                title={'text': "Risk by Department", 'x': 0.5, 'xanchor': 'center',
                      'font': {'size': 20, 'color': '#e2e8f0', 'family': 'Arial Black'}},
                barmode='stack',
                xaxis_title="Department",
                yaxis_title="Students",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(color='#e2e8f0')),
                height=400,
                paper_bgcolor='rgba(30, 41, 59, 0.7)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(size=12, color='#e2e8f0')
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Charts Row 2
        col1, col2 = st.columns(2)
        
        with col1:
            # Attendance vs Risk
            fig = go.Figure()
            for risk, color, name in [(0, '#10b981', 'Safe'), (1, '#ef4444', 'At Risk')]:
                data = df[df['dropout'] == risk]['attendance_rate']
                fig.add_trace(go.Box(y=data, name=name, marker_color=color,
                                    boxmean='sd'))
            
            fig.update_layout(
                title={'text': "Attendance Distribution by Risk", 'x': 0.5, 'xanchor': 'center',
                      'font': {'size': 20, 'color': '#e2e8f0', 'family': 'Arial Black'}},
                yaxis_title="Attendance %",
                height=400,
                paper_bgcolor='rgba(30, 41, 59, 0.7)',
                plot_bgcolor='rgba(0,0,0,0)',
                showlegend=True,
                font=dict(color='#e2e8f0'),
                legend=dict(font=dict(color='#e2e8f0'))
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Age-wise Risk
            age_data = df.groupby('age')['dropout'].agg(['sum', 'count']).reset_index()
            age_data['percentage'] = (age_data['sum'] / age_data['count']) * 100
            age_data = age_data.sort_values('age')
            
            fig = go.Figure(data=[
                go.Bar(x=age_data['age'], y=age_data['percentage'],
                      text=age_data['percentage'].round(1),
                      texttemplate='%{text}%',
                      textposition='outside',
                      marker=dict(color=age_data['percentage'],
                                 colorscale='RdYlGn_r',
                                 showscale=False,
                                 line=dict(color='white', width=2)))
            ])
            
            fig.update_layout(
                title={'text': "At-Risk % by Age", 'x': 0.5, 'xanchor': 'center',
                      'font': {'size': 20, 'color': '#e2e8f0', 'family': 'Arial Black'}},
                xaxis_title="Age",
                yaxis_title="At-Risk Percentage",
                height=400,
                paper_bgcolor='rgba(30, 41, 59, 0.7)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#e2e8f0')
            )
            st.plotly_chart(fig, use_container_width=True)
        
    else:
        st.error("⚠️ Please run the setup scripts first!")
        st.info("Run: `python generate_data.py` then `python preprocess.py` then `python train_model.py`")

# Students Page
elif page == "👥 Students":
    st.markdown('<h1 class="gradient-text fade-in">Student Management</h1>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    
    if df is not None:
        # Filters
        st.markdown('<div class="info-card"><h3 style="margin-top: 0;">Filter Students</h3></div>', unsafe_allow_html=True)
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            risk_filter = st.selectbox("🎯 Risk Status", ["All", "At Risk", "Safe"])
        with col2:
            dept_filter = st.selectbox("🏢 Department", ["All"] + sorted(df['department'].unique().tolist()))
        with col3:
            age_filter = st.selectbox("📅 Age", ["All"] + sorted(df['age'].unique().tolist()))
        with col4:
            search = st.text_input("🔍 Search", placeholder="Student ID")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Apply filters
        filtered_df = df.copy()
        if risk_filter == "At Risk":
            filtered_df = filtered_df[filtered_df['dropout'] == 1]
        elif risk_filter == "Safe":
            filtered_df = filtered_df[filtered_df['dropout'] == 0]
        
        if dept_filter != "All":
            filtered_df = filtered_df[filtered_df['department'] == dept_filter]
        if age_filter != "All":
            filtered_df = filtered_df[filtered_df['age'] == age_filter]
        if search:
            filtered_df = filtered_df[
                filtered_df['student_id'].astype(str).str.contains(search, case=False)
            ]
        
        # File Upload Section - Full Width Above Showing Students
        st.markdown("""
            <div style='background: rgba(139, 92, 246, 0.2); padding: 1rem; border-radius: 12px; border: 2px solid rgba(139, 92, 246, 0.5); margin-bottom: 1rem;'>
                <h4 style='color: #a78bfa; margin: 0 0 0.5rem 0; font-size: 1rem;'>📁 Upload New Batch</h4>
            </div>
        """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader("", type=['xlsx', 'xls', 'csv'], help="Upload Excel or CSV file with student data", label_visibility="collapsed")
        
        if uploaded_file is not None:
            model, df_default, scaler, features = load_model_and_data()
            
            if model is not None and scaler is not None and features is not None:
                # Check if file is already processed
                if 'processed_file_name' not in st.session_state or st.session_state.get('processed_file_name') != uploaded_file.name:
                    with st.spinner('Processing uploaded data...'):
                        df_processed, error, stats = process_uploaded_data(uploaded_file, model, scaler, features)
                        
                        if error:
                            st.error(f"❌ Error: {error}")
                            st.session_state.temp_processed_df = None
                            st.session_state.processed_file_name = None
                            st.session_state.upload_stats = None
                        else:
                            st.session_state.temp_processed_df = df_processed
                            st.session_state.processed_file_name = uploaded_file.name
                            st.session_state.upload_stats = stats
                            
                            # Save processed file
                            output_path = DATA_DIR / 'processed' / f'uploaded_{uploaded_file.name.replace(".xlsx", ".csv").replace(".xls", ".csv")}'
                            df_processed.to_csv(output_path, index=False)
                            
                            # Display processing information
                            st.success(f"✅ File processed successfully!")
                            
                            # Show model and processing details
                            st.markdown("""
                                <div style='background: rgba(16, 185, 129, 0.1); padding: 1rem; border-radius: 10px; border: 1px solid rgba(16, 185, 129, 0.3); margin: 1rem 0;'>
                                    <h4 style='color: #10b981; margin: 0 0 0.5rem 0;'>🤖 Model Processing Complete</h4>
                                    <p style='color: #e2e8f0; margin: 0; font-size: 0.9rem;'>
                                        <strong>Algorithm:</strong> Random Forest Classifier<br>
                                        <strong>Data Filtered:</strong> Yes (Preprocessed with feature engineering)<br>
                                        <strong>Features Used:</strong> Academic score, Engagement score, Risk indicators
                                    </p>
                                </div>
                            """, unsafe_allow_html=True)
                            
                            # Display statistics
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("📊 Total Students", f"{stats['total_students']:,}")
                            with col2:
                                st.metric("🔴 At Risk", f"{stats['at_risk']}", f"{(stats['at_risk']/stats['total_students']*100):.1f}%")
                            with col3:
                                st.metric("🟢 Safe", f"{stats['safe']}", f"{(stats['safe']/stats['total_students']*100):.1f}%")
                            
                            st.info(f"📁 Saved to: `{output_path.name}`")
                
                # Show proceed button if file is processed
                if st.session_state.get('temp_processed_df') is not None:
                    col1, col2, col3 = st.columns([1, 2, 1])
                    with col2:
                        if st.button("✨ Proceed & Update Statistics", type="primary", use_container_width=True):
                            st.session_state.uploaded_df = st.session_state.temp_processed_df
                            st.session_state.data_source = 'uploaded'
                            st.session_state.temp_processed_df = None
                            st.session_state.processed_file_name = None
                            st.success("🎉 Statistics updated with new data!")
                            st.rerun()
            else:
                st.error("Model not loaded. Please train the model first!")
        
        if st.session_state.data_source == 'uploaded' and st.session_state.uploaded_df is not None:
            col1, col2 = st.columns([4, 1])
            with col2:
                if st.button("🔄 Use Default Data", use_container_width=True):
                    st.session_state.uploaded_df = None
                    st.session_state.data_source = 'default'
                    st.rerun()
        
        st.markdown(f'<div class="info-card">Showing <strong>{len(filtered_df)}</strong> students</div>', 
                   unsafe_allow_html=True)
        
        # Display table (use 'dropout' as the target column)
        display_df = filtered_df[['student_id', 'department', 'age', 
                                   'attendance_rate', 'cgpa', 
                                   'past_failures', 'dropout']].copy()
        display_df['Status'] = display_df['dropout'].apply(lambda x: '🔴 At Risk' if x == 1 else '🟢 Safe')
        display_df = display_df.drop('dropout', axis=1)
        
        st.dataframe(
            display_df.style.format({
                'attendance_rate': '{:.1f}%',
                'cgpa': '{:.2f}',
                'past_failures': '{:.0f}'
            }),
            height=400,
            use_container_width=True
        )
        
        # Student details
        st.markdown("---")
        selected_id = st.selectbox("Select student for detailed view:", filtered_df['student_id'].tolist())
        
        if selected_id:
            student = filtered_df[filtered_df['student_id'] == selected_id].iloc[0]
            
            # Predict risk if model available
            if model is not None:
                try:
                    # This is simplified - you'd need proper preprocessing
                    risk_prob = np.random.uniform(0.2, 0.9)  # Placeholder
                    risk_cat, risk_class, risk_icon, risk_color = get_risk_category(risk_prob)
                    
                    col1, col2, col3 = st.columns([2, 1, 1])
                    
                    with col1:
                        st.markdown(f"""
                        <div class="metric-card">
                            <h2>📋 Student {student['student_id']}</h2>
                            <p style='color: #64748b;'>Department: {student['department']} | Age: {student['age']}</p>
                            <div class='{risk_class}' style='margin-top: 1rem;'>
                                {risk_icon} {risk_cat} - {risk_prob*100:.1f}%
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                        fig = create_gauge_chart(risk_prob, "Risk Score")
                        st.plotly_chart(fig, use_container_width=True)
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    with col3:
                        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                        st.metric("Attendance", f"{student['attendance_rate']:.1f}%")
                        st.metric("CGPA", f"{student['cgpa']:.2f}")
                        st.metric("Past Failures", f"{student['past_failures']:.0f}")
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                except Exception as e:
                    st.error(f"Error: {e}")

# Prediction Page  
elif page == "🎯 Prediction":
    st.markdown('<h1 class="gradient-text fade-in">Risk Prediction Simulator</h1>', unsafe_allow_html=True)
    
    st.markdown('<div class="info-card">Adjust parameters to simulate student profiles and predict risk levels</div>', 
               unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.subheader("📚 Academic Factors")
        attendance = st.slider("Attendance %", 0, 100, 75)
        cgpa = st.slider("CGPA", 0.0, 10.0, 7.0, 0.1)
        assignments = st.slider("Assignments Submitted", 0, 100, 80)
        past_failures = st.slider("Past Failures", 0, 5, 0)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.subheader("👤 Personal & Support")
        study_hours = st.slider("Study Hours/Week", 0, 40, 20)
        projects = st.slider("Projects Completed", 0, 10, 5)
        sports = st.selectbox("Sports Participation", ["Yes", "No"])
        scholarship = st.selectbox("Scholarship", ["Yes", "No"])
        st.markdown('</div>', unsafe_allow_html=True)
    
    if st.button("🎯 Predict Risk", type="primary"):
        # Simplified prediction logic
        sports_val = 1 if sports == "Yes" else 0
        scholarship_val = 1 if scholarship == "Yes" else 0
        
        risk_score = (
            (100 - attendance) * 0.25 +
            (10 - cgpa) * 10 * 0.20 +
            past_failures * 15 * 0.15 +
            (100 - assignments) * 0.10 +
            (40 - study_hours) * 2 * 0.10 +
            (1 - sports_val) * 10 * 0.10 +
            (1 - scholarship_val) * 10 * 0.10
        ) / 100
        
        risk_score = min(max(risk_score, 0), 1)
        risk_cat, risk_class, risk_icon, risk_color = get_risk_category(risk_score)
        
        st.markdown("---")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Risk Score", f"{risk_score*100:.1f}%")
            st.markdown(f'<div class="{risk_class}">{risk_icon} {risk_cat}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            fig = create_gauge_chart(risk_score)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Status", "At Risk" if risk_score >= 0.6 else "Safe")
            st.markdown('</div>', unsafe_allow_html=True)

# Analytics Page
elif page == "📊 Analytics":
    st.markdown('<h1 class="gradient-text fade-in">Advanced Analytics</h1>', unsafe_allow_html=True)
    
    if df is not None:
        tab1, tab2, tab3 = st.tabs(["📈 Trends", "🔗 Correlations", "🎯 Feature Impact"])
        
        with tab1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.subheader("Department Statistics")
            dept_stats = df.groupby('department').agg({
                'dropout': ['sum', 'count', 'mean']
            }).round(3)
            dept_stats.columns = ['At Risk', 'Total', 'Risk Rate']
            st.dataframe(dept_stats, width="stretch")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with tab2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            col1, col2 = st.columns(2)
            with col1:
                feat1 = st.selectbox("Feature 1", numeric_cols, index=0)
            with col2:
                feat2 = st.selectbox("Feature 2", numeric_cols, index=1)
            
            fig = px.scatter(df, x=feat1, y=feat2, color='dropout',
                           color_discrete_map={0: '#10b981', 1: '#ef4444'},
                           title=f"{feat1} vs {feat2}")
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with tab3:
            st.markdown('<div class="info-card">Feature importance analysis coming soon...</div>',
                       unsafe_allow_html=True)

# AI Assistant Page
elif page == "💬 AI Assistant":
    st.markdown('<h1 class="gradient-text fade-in">AI Assistant</h1>', unsafe_allow_html=True)
    
    st.markdown('<div class="info-card">Ask me anything about student analytics, interventions, or system navigation!</div>',
               unsafe_allow_html=True)
    
    if 'messages' not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "👋 Hello! I'm your AI assistant. How can I help you today?"}
        ]
    
    # Display chat
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(f'<div class="chat-message">{msg["content"]}</div>', unsafe_allow_html=True)
    
    # Chat input
    if prompt := st.chat_input("Type your message..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Simple response logic
        response = "I'm here to help! You can ask about:\n- System navigation\n- Intervention strategies\n- Student statistics\n- Risk assessment"
        
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.rerun()

# Footer
st.markdown("---")
