"""Generate synthetic student data"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from config import RAW_DATA_PATH

np.random.seed(42)
random.seed(42)

print("🎓 Generating student data...")

num_students = 1000
data = {
    'student_id': [f'STU{str(i).zfill(5)}' for i in range(1, num_students + 1)],
    'name': [f'Student_{i}' for i in range(1, num_students + 1)],
    'age': np.random.randint(17, 26, num_students),
    'gender': np.random.choice(['Male', 'Female'], num_students),
    'department': np.random.choice(['Computer Science', 'Mechanical', 'Civil', 'Electrical', 'Electronics'], num_students),
    'year': np.random.choice([1, 2, 3, 4], num_students),
    'semester': np.random.choice([1, 2], num_students),
    'attendance_percentage': np.random.beta(8, 2, num_students) * 100,
    'internal_marks': np.random.beta(5, 2, num_students) * 100,
    'previous_semester_gpa': np.random.beta(6, 2, num_students) * 10,
    'assignments_submitted': np.random.randint(0, 11, num_students),
    'total_assignments': 10,
    'test_scores_avg': np.random.beta(5, 2, num_students) * 100,
    'failed_subjects': np.random.choice([0, 1, 2], num_students, p=[0.7, 0.25, 0.05]),
    'library_visits': np.random.poisson(15, num_students),
    'counseling_sessions': np.random.choice([0, 1, 2, 3], num_students, p=[0.5, 0.3, 0.15, 0.05]),
    'participation_score': np.random.beta(5, 3, num_students) * 10,
    'late_submissions': np.random.poisson(2, num_students),
    'disciplinary_actions': np.random.choice([0, 1], num_students, p=[0.9, 0.1]),
    'extracurricular_activities': np.random.choice([0, 1, 2, 3], num_students, p=[0.3, 0.4, 0.2, 0.1]),
    'financial_aid': np.random.choice([0, 1], num_students, p=[0.4, 0.6]),
    'commute_time': np.random.choice([15, 30, 45, 60, 90], num_students),
    'family_support_index': np.random.beta(6, 2, num_students) * 10,
    'work_hours_per_week': np.random.choice([0, 5, 10, 15, 20], num_students, p=[0.5, 0.2, 0.15, 0.1, 0.05]),
}

df = pd.DataFrame(data)

# Generate risk
risk_score = (
    (100 - df['attendance_percentage']) * 0.15 +
    (100 - df['internal_marks']) * 0.10 +
    (10 - df['previous_semester_gpa']) * 5 * 0.12 +
    (10 - df['assignments_submitted']) * 3 * 0.08 +
    df['failed_subjects'] * 15 * 0.10 +
    (1 - df['financial_aid']) * 10 * 0.08
)

risk_score = risk_score / risk_score.max()
risk_score = risk_score + np.random.normal(0, 0.05, num_students)
risk_score = np.clip(risk_score, 0, 1)

# 'disengaged' legacy column removed; use 'dropout' as canonical target
df['dropout'] = (risk_score > 0.55).astype(int)
df['risk_score'] = risk_score

output_path = RAW_DATA_PATH / 'student_data.csv'
df.to_csv(output_path, index=False)

print(f"✓ Generated {len(df)} student records")
print(f"✓ At-risk students: {df['dropout'].sum()} ({df['dropout'].mean()*100:.1f}%)")
print(f"✓ Saved to: {output_path}")
