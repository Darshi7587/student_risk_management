"""
Test script to verify prediction accuracy between training data and new predictions
"""
import pandas as pd
import numpy as np
import joblib
from config import *

print("🔍 Testing Prediction Accuracy\n" + "="*60)

# Load trained model and preprocessed data
print("\n1️⃣ Loading model and original training data...")
model = joblib.load(MODEL_FILE)
scaler = joblib.load(SCALER_FILE)
features = joblib.load(FEATURE_NAMES_FILE)
df_original = pd.read_csv(CLEANED_DATA_FILE)

print(f"   ✓ Model loaded: {type(model).__name__}")
print(f"   ✓ Features expected: {len(features)}")
print(f"   ✓ Feature names: {features}")

# Take a sample of original data and re-predict
print("\n2️⃣ Testing predictions on original training data sample...")
sample_size = 10
df_sample = df_original.sample(n=sample_size, random_state=42)

# Extract features
X_sample = df_sample[features]
X_sample_scaled = scaler.transform(X_sample)

# Make predictions
predictions = model.predict(X_sample_scaled)
pred_proba = model.predict_proba(X_sample_scaled)[:, 1]

# Compare with original labels
original_labels = df_sample['dropout'].values

print(f"\n   Comparing predictions vs original labels:")
print(f"   {'Student ID':<15} {'Original':<12} {'Predicted':<12} {'Probability':<12} {'Match'}")
print(f"   {'-'*65}")

matches = 0
for i, (idx, row) in enumerate(df_sample.iterrows()):
    student_id = row['student_id']
    orig = int(original_labels[i])
    pred = int(predictions[i])
    prob = pred_proba[i]
    match = "✓" if orig == pred else "✗"
    if orig == pred:
        matches += 1
    print(f"   {student_id:<15} {orig:<12} {pred:<12} {prob:<12.4f} {match}")

accuracy = matches / sample_size * 100
print(f"\n   Sample Accuracy: {accuracy:.1f}% ({matches}/{sample_size})")

# Now test with raw data (simulate uploading)
print("\n3️⃣ Testing with 'uploaded' raw data (preprocessing from scratch)...")
print("   Reading raw dataset...")

# Read the original raw dataset
try:
    df_raw = pd.read_csv(r"c:\Users\darsh\Downloads\Dataset - Dataset.csv")
    print(f"   ✓ Raw data loaded: {len(df_raw)} records")
except:
    df_raw = pd.read_csv(RAW_DATA_FILE)
    print(f"   ✓ Raw data loaded from backup: {len(df_raw)} records")

# Take same students as sample
test_student_ids = df_sample['student_id'].tolist()
df_raw_sample = df_raw[df_raw['student_id'].isin(test_student_ids)].copy()

print(f"   ✓ Extracted {len(df_raw_sample)} matching students from raw data")

# Apply EXACT preprocessing from preprocess.py
print("\n   Applying preprocessing...")

# Standardize gender
df_raw_sample['gender'] = df_raw_sample['gender'].astype(str).str.upper().replace({'M': 'Male', 'FEMALE': 'Female', 'NA': 'Other', 'NAN': 'Other'})
df_raw_sample['gender'] = df_raw_sample['gender'].fillna('Other')

# Standardize department
df_raw_sample['department'] = df_raw_sample['department'].astype(str).str.upper()
df_raw_sample['department'] = df_raw_sample['department'].fillna('UNKNOWN')

# Standardize scholarship
df_raw_sample['scholarship'] = df_raw_sample['scholarship'].astype(str).str.upper().replace({'Y': 'Yes', 'N': 'No', 'NOPE': 'No', 'NAN': 'No'})
df_raw_sample['scholarship'] = df_raw_sample['scholarship'].fillna('No')
df_raw_sample['scholarship_binary'] = (df_raw_sample['scholarship'] == 'Yes').astype(int)

# Standardize sports
df_raw_sample['sports_participation'] = df_raw_sample['sports_participation'].astype(str).str.upper().replace({'Y': 'Yes', 'N': 'No', 'NAN': 'No'})
df_raw_sample['sports_participation'] = df_raw_sample['sports_participation'].fillna('No')
df_raw_sample['sports_binary'] = (df_raw_sample['sports_participation'] == 'Yes').astype(int)

# Parental education
df_raw_sample['parental_education'] = df_raw_sample['parental_education'].astype(str).fillna('Unknown')
df_raw_sample['parental_education'] = df_raw_sample['parental_education'].replace({'nan': 'Unknown', 'NA': 'Unknown', 'NAN': 'Unknown'})

# Numeric columns
numeric_cols = ['age', 'cgpa', 'attendance_rate', 'family_income', 'past_failures', 
                'study_hours_per_week', 'assignments_submitted', 'projects_completed', 'extra_curricular']
for col in numeric_cols:
    if col in df_raw_sample.columns:
        df_raw_sample[col] = pd.to_numeric(df_raw_sample[col], errors='coerce')
        df_raw_sample[col] = df_raw_sample[col].fillna(df_raw_sample[col].median())

# Handle outliers
df_raw_sample['family_income'] = df_raw_sample['family_income'].abs()

# Feature engineering
df_raw_sample['total_activities'] = df_raw_sample['extra_curricular'] + df_raw_sample['sports_binary']
df_raw_sample['academic_score'] = (df_raw_sample['cgpa'] / 10 * 50) + (df_raw_sample['attendance_rate'] / 100 * 30) + (df_raw_sample['assignments_submitted'] / 60 * 20)
df_raw_sample['engagement_score'] = (df_raw_sample['total_activities'] * 2) + (df_raw_sample['projects_completed'] * 3) + (df_raw_sample['sports_binary'] * 5)

# Risk indicators
df_raw_sample['low_attendance'] = (df_raw_sample['attendance_rate'] < 70).astype(int)
df_raw_sample['low_cgpa'] = (df_raw_sample['cgpa'] < 5.0).astype(int)
df_raw_sample['high_failures'] = (df_raw_sample['past_failures'] > 3).astype(int)
df_raw_sample['low_study_hours'] = (df_raw_sample['study_hours_per_week'] < 10).astype(int)
df_raw_sample['low_income'] = (df_raw_sample['family_income'] < df_raw_sample['family_income'].median()).astype(int)

# Load and apply saved label encoders
label_encoders = joblib.load(MODEL_DIR / 'label_encoders.pkl')

known_genders = set(label_encoders['gender'].classes_)
df_raw_sample['gender'] = df_raw_sample['gender'].apply(lambda x: x if x in known_genders else 'Other')
df_raw_sample['gender_encoded'] = label_encoders['gender'].transform(df_raw_sample['gender'])

known_depts = set(label_encoders['department'].classes_)
df_raw_sample['department'] = df_raw_sample['department'].apply(lambda x: x if x in known_depts else 'UNKNOWN')
df_raw_sample['department_encoded'] = label_encoders['department'].transform(df_raw_sample['department'])

known_edu = set(label_encoders['parental_education'].classes_)
df_raw_sample['parental_education'] = df_raw_sample['parental_education'].apply(lambda x: x if x in known_edu else 'Unknown')
df_raw_sample['parental_education_encoded'] = label_encoders['parental_education'].transform(df_raw_sample['parental_education'])

print("   ✓ Preprocessing complete")

# Make predictions
print("\n   Making predictions...")
X_raw = df_raw_sample[features]
X_raw_scaled = scaler.transform(X_raw)
predictions_raw = model.predict(X_raw_scaled)
pred_proba_raw = model.predict_proba(X_raw_scaled)[:, 1]

print(f"\n   Comparing raw data predictions vs original labels:")
print(f"   {'Student ID':<15} {'Original':<12} {'New Pred':<12} {'Probability':<12} {'Match'}")
print(f"   {'-'*65}")

matches_raw = 0
for i, (idx, row) in enumerate(df_raw_sample.iterrows()):
    student_id = row['student_id']
    # Get original label from df_sample
    orig = int(df_sample[df_sample['student_id'] == student_id]['dropout'].values[0])
    pred = int(predictions_raw[i])
    prob = pred_proba_raw[i]
    match = "✓" if orig == pred else "✗"
    if orig == pred:
        matches_raw += 1
    print(f"   {student_id:<15} {orig:<12} {pred:<12} {prob:<12.4f} {match}")

accuracy_raw = matches_raw / len(df_raw_sample) * 100
print(f"\n   Raw Data Accuracy: {accuracy_raw:.1f}% ({matches_raw}/{len(df_raw_sample)})")

print("\n" + "="*60)
print("✅ Test Complete!")
print(f"\nSummary:")
print(f"  • Processed data accuracy: {accuracy:.1f}%")
print(f"  • Raw data accuracy: {accuracy_raw:.1f}%")
if accuracy_raw >= 90:
    print(f"  • Status: ✅ EXCELLENT - Predictions are consistent")
elif accuracy_raw >= 70:
    print(f"  • Status: ⚠️  GOOD - Minor discrepancies detected")
else:
    print(f"  • Status: ❌ POOR - Significant preprocessing differences")
print("="*60)
