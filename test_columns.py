"""Quick test to verify columns match"""
import pandas as pd
from config import *

print("Testing data compatibility...")
print("="*60)

# Load data
df = pd.read_csv(CLEANED_DATA_FILE)
print(f"\n✓ Loaded {len(df):,} records")

# Check required columns
required_cols = [
    'student_id', 'department', 'age', 'attendance_rate', 'cgpa',
    'past_failures', 'disengaged', 'study_hours_per_week',
    'assignments_submitted', 'projects_completed', 'sports_participation'
]

print(f"\nChecking required columns...")
missing = []
for col in required_cols:
    if col in df.columns:
        print(f"  ✓ {col}")
    else:
        print(f"  ✗ {col} - MISSING!")
        missing.append(col)

if missing:
    print(f"\n❌ Missing columns: {missing}")
    print("\nAvailable columns:")
    print(df.columns.tolist())
else:
    print(f"\n✅ All required columns present!")
    print(f"\nDataset summary:")
    print(f"  Total students: {len(df):,}")
    print(f"  At risk: {df['disengaged'].sum():,} ({df['disengaged'].mean()*100:.1f}%)")
    print(f"  Avg attendance: {df['attendance_rate'].mean():.1f}%")
    print(f"  Avg CGPA: {df['cgpa'].mean():.2f}")
    print(f"  Departments: {df['department'].nunique()}")
    print(f"  Age range: {df['age'].min():.0f} - {df['age'].max():.0f}")

print("\n" + "="*60)
print("Test complete!")
