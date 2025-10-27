"""Configuration settings for Student Disengagement Prediction System"""
import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).parent

# Data paths
DATA_DIR = BASE_DIR / 'data'
RAW_DATA_PATH = DATA_DIR / 'raw'
PROCESSED_DATA_PATH = DATA_DIR / 'processed'
CLEANED_DATA_FILE = PROCESSED_DATA_PATH / 'cleaned_student_data.csv'

# Model paths
MODEL_DIR = BASE_DIR / 'models'
MODEL_FILE = MODEL_DIR / 'student_model.pkl'
SCALER_FILE = MODEL_DIR / 'scaler.pkl'
FEATURE_NAMES_FILE = MODEL_DIR / 'feature_names.pkl'

# Reports
REPORTS_DIR = BASE_DIR / 'reports'
FIGURES_DIR = REPORTS_DIR / 'figures'

# Risk thresholds
RISK_THRESHOLDS = {'low': 0.3, 'medium': 0.6, 'high': 1.0}

# Create directories
for directory in [DATA_DIR, RAW_DATA_PATH, PROCESSED_DATA_PATH, MODEL_DIR, REPORTS_DIR, FIGURES_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Intervention strategies
INTERVENTIONS = {
    'high_risk': {
        'academic': ['Immediate academic counseling', 'Personalized tutoring', 'Weekly monitoring', 'Remedial classes'],
        'financial': ['Emergency financial aid', 'Scholarship review', 'Campus job assistance'],
        'behavioral': ['Mental health counseling', 'Peer mentorship', 'Study skills workshop']
    },
    'medium_risk': {
        'academic': ['Bi-weekly check-ins', 'Study groups', 'Time management workshop'],
        'support': ['Financial planning', 'Campus resources orientation']
    },
    'low_risk': {
        'support': ['Continue monitoring', 'Excellence programs', 'Leadership opportunities']
    }
}
