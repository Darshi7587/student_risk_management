"""Data preprocessing pipeline for real student dataset"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
from config import *

print("🔧 Starting data preprocessing...")

# Load the real dataset
df = pd.read_csv(r"c:\Users\darsh\Downloads\Dataset - Dataset.csv")
print(f"✓ Loaded {len(df)} records")

# Data cleaning
print("Cleaning data...")

# Handle missing values
# Standardize gender
df['gender'] = df['gender'].str.upper().replace({'M': 'Male', 'FEMALE': 'Female', 'NA': 'Other'})
df['gender'] = df['gender'].fillna('Other')

# Standardize department
df['department'] = df['department'].str.upper()
df['department'] = df['department'].fillna('UNKNOWN')

# Standardize scholarship
df['scholarship'] = df['scholarship'].str.upper().replace({'Y': 'Yes', 'N': 'No', 'NOPE': 'No', 'NAN': 'No'})
df['scholarship'] = df['scholarship'].fillna('No')
df['scholarship_binary'] = (df['scholarship'] == 'Yes').astype(int)

# Standardize sports participation
df['sports_participation'] = df['sports_participation'].str.upper().replace({'Y': 'Yes', 'N': 'No', 'NAN': 'No'})
df['sports_participation'] = df['sports_participation'].fillna('No')
df['sports_binary'] = (df['sports_participation'] == 'Yes').astype(int)

# Handle parental education
df['parental_education'] = df['parental_education'].fillna('Unknown')
df['parental_education'] = df['parental_education'].replace({'nan': 'Unknown', 'NA': 'Unknown'})

# Fill numeric missing values with median
numeric_columns = ['age', 'cgpa', 'attendance_rate', 'family_income', 'past_failures', 
                   'study_hours_per_week', 'assignments_submitted', 'projects_completed', 
                   'total_activities']

for col in numeric_columns:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col] = df[col].fillna(df[col].median())

# Handle outliers in family_income (negative values)
df['family_income'] = df['family_income'].abs()

# Remove duplicates
df = df.drop_duplicates(subset=['student_id'], keep='first')
print(f"✓ Cleaned data: {len(df)} records")

# Feature Engineering
print("Creating engineered features...")

# Academic performance index
df['academic_score'] = (df['cgpa'] / 10 * 50) + (df['attendance_rate'] / 100 * 30) + (df['assignments_submitted'] / 60 * 20)

# Engagement score
df['engagement_score'] = (df['total_activities'] * 2) + (df['projects_completed'] * 3) + (df['sports_binary'] * 5)

# Risk indicators
df['low_attendance'] = (df['attendance_rate'] < 70).astype(int)
df['low_cgpa'] = (df['cgpa'] < 5.0).astype(int)
df['high_failures'] = (df['past_failures'] > 3).astype(int)
df['low_study_hours'] = (df['study_hours_per_week'] < 10).astype(int)

# Financial stress indicator
df['low_income'] = (df['family_income'] < df['family_income'].median()).astype(int)

print(f"✓ Created {5} engineered features")

# Encode categorical variables
label_encoders = {}

categorical_cols = ['gender', 'department', 'parental_education']
for col in categorical_cols:
    le = LabelEncoder()
    df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

print(f"✓ Encoded categorical features")

# Prepare target variable (dropout)
df['disengaged'] = df['dropout']  # Rename for consistency

# Save cleaned data
df.to_csv(CLEANED_DATA_FILE, index=False)
print(f"✓ Saved cleaned data to: {CLEANED_DATA_FILE}")

# Prepare features for modeling
exclude_cols = ['student_id', 'dropout', 'disengaged', 'gender', 'department', 
                'scholarship', 'parental_education', 'extra_curricular', 
                'sports_participation']

feature_cols = [col for col in df.columns if col not in exclude_cols]

X = df[feature_cols]
y = df['disengaged']

# Scale numeric features
scaler = StandardScaler()
numeric_cols = X.select_dtypes(include=[np.number]).columns
X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

# Save preprocessing artifacts
joblib.dump(scaler, SCALER_FILE)
joblib.dump(feature_cols, FEATURE_NAMES_FILE)
joblib.dump(label_encoders, MODEL_DIR / 'label_encoders.pkl')
print(f"✓ Saved preprocessing artifacts")

print(f"\n{'='*60}")
print(f"✅ Preprocessing Complete!")
print(f"{'='*60}")
print(f"Total Students:     {len(df):,}")
print(f"Features:           {len(feature_cols)}")
print(f"Dropout Students:   {y.sum():,} ({y.mean()*100:.1f}%)")
print(f"Safe Students:      {(~y.astype(bool)).sum():,} ({(1-y.mean())*100:.1f}%)")
print(f"{'='*60}\n")
