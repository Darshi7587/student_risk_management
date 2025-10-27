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
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3), 
                    inset 0 1px 0 rgba(255, 255, 255, 0.1);
        border: 1px solid rgba(139, 92, 246, 0.2);
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        margin: 1rem 0;
    }
    
    .metric-card:hover {
        transform: translateY(-8px) scale(1.02);
        box-shadow: 0 16px 48px rgba(139, 92, 246, 0.3);
        border-color: rgba(139, 92, 246, 0.5);
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
        color: #1e293b;
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
        background: rgba(30, 41, 59, 0.5) !important;
        border: 2px solid rgba(139, 92, 246, 0.3) !important;
        border-radius: 10px !important;
        color: #e2e8f0 !important;
        padding: 0.75rem !important;
        font-size: 1rem;
    }
    
    .stTextInput>div>div>input:focus,
    .stSelectbox>div>div>div:focus {
        border-color: #8b5cf6 !important;
        box-shadow: 0 0 0 3px rgba(139, 92, 246, 0.2) !important;
    }
    
    /* Labels */
    label {
        color: #e2e8f0 !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
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
        title={'text': title, 'font': {'size': 24, 'color': '#1e293b', 'family': 'Arial Black'}},
        number={'suffix': "%", 'font': {'size': 40, 'color': '#1e293b'}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 2, 'tickcolor': "#cbd5e1"},
            'bar': {'color': "#667eea", 'thickness': 0.75},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "#e2e8f0",
            'steps': [
                {'range': [0, 30], 'color': 'rgba(16, 185, 129, 0.2)'},
                {'range': [30, 60], 'color': 'rgba(245, 158, 11, 0.2)'},
                {'range': [60, 100], 'color': 'rgba(239, 68, 68, 0.2)'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 60
            }
        }
    ))
    
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={'family': "Arial"},
        height=300,
        margin=dict(l=20, r=20, t=60, b=20)
    )
    return fig

# ==================== SIDEBAR ====================

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
    
    model, df, scaler, features = load_model_and_data()
    
    if model is not None:
        st.success("✓ Model Active")
    else:
        st.error("✗ Model Not Found")
    
    if df is not None:
        st.success(f"✓ {len(df)} Students Loaded")
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
        
        at_risk = df['disengaged'].sum() if 'disengaged' in df.columns else 0
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
        at_risk = df['disengaged'].sum()
        avg_attendance = df['attendance_rate'].mean()
        avg_gpa = df['cgpa'].mean()
        
        with col1:
            st.markdown('<div class="metric-card fade-in">', unsafe_allow_html=True)
            st.metric("Total Students", f"{total:,}", help="Total enrolled students")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-card fade-in">', unsafe_allow_html=True)
            st.metric("At Risk", f"{at_risk}", f"{(at_risk/total*100):.1f}%", delta_color="inverse")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="metric-card fade-in">', unsafe_allow_html=True)
            st.metric("Safe", f"{total-at_risk}", f"{((total-at_risk)/total*100):.1f}%")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col4:
            st.markdown('<div class="metric-card fade-in">', unsafe_allow_html=True)
            st.metric("Avg Attendance", f"{avg_attendance:.1f}%")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col5:
            st.markdown('<div class="metric-card fade-in">', unsafe_allow_html=True)
            st.metric("Avg GPA", f"{avg_gpa:.2f}/10")
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Charts Row 1
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            # Risk Distribution Pie
            risk_counts = df['disengaged'].value_counts()
            fig = go.Figure(data=[go.Pie(
                labels=['Safe', 'At Risk'],
                values=[risk_counts.get(0, 0), risk_counts.get(1, 0)],
                hole=0.5,
                marker=dict(colors=['#10b981', '#ef4444'], 
                           line=dict(color='white', width=3)),
                textfont=dict(size=16, color='white', family='Arial Black'),
                pull=[0.05, 0.1]
            )])
            fig.update_layout(
                title={'text': "Student Risk Distribution", 'x': 0.5, 'xanchor': 'center',
                      'font': {'size': 20, 'color': '#1e293b', 'family': 'Arial Black'}},
                showlegend=True,
                legend=dict(font=dict(size=14)),
                height=400,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig, width="stretch")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            # Department Analysis
            dept_data = df.groupby('department')['disengaged'].agg(['sum', 'count']).reset_index()
            dept_data['safe'] = dept_data['count'] - dept_data['sum']
            
            fig = go.Figure()
            fig.add_trace(go.Bar(name='At Risk', x=dept_data['department'], y=dept_data['sum'],
                                marker_color='#ef4444', text=dept_data['sum'], textposition='auto'))
            fig.add_trace(go.Bar(name='Safe', x=dept_data['department'], y=dept_data['safe'],
                                marker_color='#10b981', text=dept_data['safe'], textposition='auto'))
            
            fig.update_layout(
                title={'text': "Risk by Department", 'x': 0.5, 'xanchor': 'center',
                      'font': {'size': 20, 'color': '#1e293b', 'family': 'Arial Black'}},
                barmode='stack',
                xaxis_title="Department",
                yaxis_title="Students",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                height=400,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(size=12)
            )
            st.plotly_chart(fig, width="stretch")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Charts Row 2
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            # Attendance vs Risk
            fig = go.Figure()
            for risk, color, name in [(0, '#10b981', 'Safe'), (1, '#ef4444', 'At Risk')]:
                data = df[df['disengaged'] == risk]['attendance_rate']
                fig.add_trace(go.Box(y=data, name=name, marker_color=color,
                                    boxmean='sd'))
            
            fig.update_layout(
                title={'text': "Attendance Distribution by Risk", 'x': 0.5, 'xanchor': 'center',
                      'font': {'size': 20, 'color': '#1e293b', 'family': 'Arial Black'}},
                yaxis_title="Attendance %",
                height=400,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                showlegend=True
            )
            st.plotly_chart(fig, width="stretch")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            # Age-wise Risk
            age_data = df.groupby('age')['disengaged'].agg(['sum', 'count']).reset_index()
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
                      'font': {'size': 20, 'color': '#1e293b', 'family': 'Arial Black'}},
                xaxis_title="Age",
                yaxis_title="At-Risk Percentage",
                height=400,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig, width="stretch")
            st.markdown('</div>', unsafe_allow_html=True)
        
    else:
        st.error("⚠️ Please run the setup scripts first!")
        st.info("Run: `python generate_data.py` then `python preprocess.py` then `python train_model.py`")

# Students Page
elif page == "👥 Students":
    st.markdown('<h1 class="gradient-text fade-in">Student Management</h1>', unsafe_allow_html=True)
    
    if df is not None:
        # Filters
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            risk_filter = st.selectbox("🎯 Risk Status", ["All", "At Risk", "Safe"])
        with col2:
            dept_filter = st.selectbox("🏢 Department", ["All"] + sorted(df['department'].unique().tolist()))
        with col3:
            age_filter = st.selectbox("📅 Age", ["All"] + sorted(df['age'].unique().tolist()))
        with col4:
            search = st.text_input("🔍 Search", placeholder="Student ID")
        
        # Apply filters
        filtered_df = df.copy()
        if risk_filter == "At Risk":
            filtered_df = filtered_df[filtered_df['disengaged'] == 1]
        elif risk_filter == "Safe":
            filtered_df = filtered_df[filtered_df['disengaged'] == 0]
        
        if dept_filter != "All":
            filtered_df = filtered_df[filtered_df['department'] == dept_filter]
        if age_filter != "All":
            filtered_df = filtered_df[filtered_df['age'] == age_filter]
        if search:
            filtered_df = filtered_df[
                filtered_df['student_id'].astype(str).str.contains(search, case=False)
            ]
        
        st.markdown(f'<div class="info-card">Showing <strong>{len(filtered_df)}</strong> students</div>', 
                   unsafe_allow_html=True)
        
        # Display table
        display_df = filtered_df[['student_id', 'department', 'age', 
                                   'attendance_rate', 'cgpa', 
                                   'past_failures', 'disengaged']].copy()
        display_df['Status'] = display_df['disengaged'].apply(lambda x: '🔴 At Risk' if x == 1 else '🟢 Safe')
        display_df = display_df.drop('disengaged', axis=1)
        
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.dataframe(
            display_df.style.format({
                'attendance_rate': '{:.1f}%',
                'cgpa': '{:.2f}',
                'past_failures': '{:.0f}'
            }),
            height=400,
            width="stretch"
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
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
                        st.plotly_chart(fig, width="stretch")
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
            st.plotly_chart(fig, width="stretch")
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
                'disengaged': ['sum', 'count', 'mean']
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
            
            fig = px.scatter(df, x=feat1, y=feat2, color='disengaged',
                           color_discrete_map={0: '#10b981', 1: '#ef4444'},
                           title=f"{feat1} vs {feat2}")
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, width="stretch")
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
st.markdown("""
    <div style='text-align: center; color: white; padding: 2rem;'>
        <p style='font-size: 0.9rem;'>Built with ❤️ for Student Success | Powered by AI</p>
    </div>
""", unsafe_allow_html=True)
