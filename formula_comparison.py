"""
Compare old vs new preprocessing formulas
"""
import pandas as pd

# Sample student data
data = {
    'cgpa': 4.08,
    'attendance_rate': 75.89,
    'assignments_submitted': 51,
    'extra_curricular': 5,
    'sports_binary': 0,
    'projects_completed': 0,
    'study_hours_per_week': 22.4,
    'past_failures': 4,
    'family_income': 32131
}

print("="*70)
print("PREPROCESSING FORMULA COMPARISON")
print("="*70)

print("\n📊 Sample Student Data:")
for key, val in data.items():
    print(f"   {key:25} = {val}")

print("\n" + "-"*70)
print("1️⃣ ACADEMIC_SCORE")
print("-"*70)

old_academic = (data['cgpa'] / 10) * 0.4 + (data['attendance_rate'] / 100) * 0.3 + (data['assignments_submitted'] / 100) * 0.3
new_academic = (data['cgpa'] / 10 * 50) + (data['attendance_rate'] / 100 * 30) + (data['assignments_submitted'] / 60 * 20)

print(f"   OLD (WRONG): {old_academic:.6f}")
print(f"   Calculation: (4.08/10)*0.4 + (75.89/100)*0.3 + (51/100)*0.3")
print(f"               = 0.1632 + 0.2277 + 0.153 = {old_academic:.4f}")
print()
print(f"   NEW (CORRECT): {new_academic:.6f}")
print(f"   Calculation: (4.08/10*50) + (75.89/100*30) + (51/60*20)")
print(f"               = 20.4 + 22.767 + 17.0 = {new_academic:.4f}")
print()
print(f"   ❌ Difference: {abs(new_academic - old_academic):.4f} (HUGE!)")

print("\n" + "-"*70)
print("2️⃣ ENGAGEMENT_SCORE")
print("-"*70)

total_activities = data['extra_curricular'] + data['sports_binary']

old_engagement = (total_activities / 2) * 0.3 + (data['study_hours_per_week'] / 40) * 0.4 + (data['projects_completed'] / 10) * 0.3
new_engagement = (total_activities * 2) + (data['projects_completed'] * 3) + (data['sports_binary'] * 5)

print(f"   OLD (WRONG): {old_engagement:.6f}")
print(f"   Calculation: (5/2)*0.3 + (22.4/40)*0.4 + (0/10)*0.3")
print(f"               = 0.75 + 0.224 + 0.0 = {old_engagement:.4f}")
print()
print(f"   NEW (CORRECT): {new_engagement:.6f}")
print(f"   Calculation: (5*2) + (0*3) + (0*5)")
print(f"               = 10 + 0 + 0 = {new_engagement:.4f}")
print()
print(f"   ❌ Difference: {abs(new_engagement - old_engagement):.4f} (HUGE!)")

print("\n" + "-"*70)
print("3️⃣ RISK INDICATORS")
print("-"*70)

print("\n   low_attendance:")
print(f"      OLD: attendance < 75 → {data['attendance_rate']} < 75 = {data['attendance_rate'] < 75}")
print(f"      NEW: attendance < 70 → {data['attendance_rate']} < 70 = {data['attendance_rate'] < 70}")

print("\n   low_cgpa:")
print(f"      OLD: cgpa < 6.0 → {data['cgpa']} < 6.0 = {data['cgpa'] < 6.0}")
print(f"      NEW: cgpa < 5.0 → {data['cgpa']} < 5.0 = {data['cgpa'] < 5.0}")

print("\n   high_failures:")
print(f"      OLD: failures >= 2 → {data['past_failures']} >= 2 = {data['past_failures'] >= 2}")
print(f"      NEW: failures > 3 → {data['past_failures']} > 3 = {data['past_failures'] > 3}")

print("\n   low_study_hours:")
print(f"      OLD: study_hours < 15 → {data['study_hours_per_week']} < 15 = {data['study_hours_per_week'] < 15}")
print(f"      NEW: study_hours < 10 → {data['study_hours_per_week']} < 10 = {data['study_hours_per_week'] < 10}")

print("\n" + "="*70)
print("🎯 CONCLUSION")
print("="*70)
print("""
The formulas were COMPLETELY DIFFERENT!

This explains why predictions were inaccurate:
- Academic scores were off by ~40 points
- Engagement scores were off by ~9 points  
- Risk indicators had different thresholds

The model was trained with the NEW (correct) formulas from preprocess.py,
but app.py was using the OLD (wrong) formulas during prediction.

✅ NOW FIXED: app.py uses the exact same formulas as preprocess.py
""")
print("="*70)
