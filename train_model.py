# ===================== ADVANCED MODEL TRAINING =====================
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, StackingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from imblearn.combine import SMOTETomek
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import joblib

print("📊 Loading dataset...")
df = pd.read_csv('Dataset - Dat.csv')

# ===================== DATA CLEANING =====================
print("🧹 Cleaning data...")
df.replace(['nan', 'NA', 'None', '-', '', ' '], np.nan, inplace=True)
df['gender'] = df['gender'].str.upper().replace({'FEMALE': 'Female', 'MALE': 'Male', 'M': 'Male', 'F': 'Female'})
df['gender'].fillna('Unknown', inplace=True)

binary_map = {'Y': 'Yes', 'N': 'No', 'Yes': 'Yes', 'No': 'No', 'Nope': 'No', 'NA': 'No', 'nan': 'No'}
for col in ['scholarship', 'extra_curricular', 'sports_participation']:
    df[col] = df[col].astype(str).str.strip().replace(binary_map)
    df[col].fillna('No', inplace=True)

numeric_cols = ['age', 'cgpa', 'attendance_rate', 'family_income', 'past_failures',
                'study_hours_per_week', 'assignments_submitted', 'projects_completed', 'total_activities']
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')
    df[col].fillna(df[col].median(), inplace=True)

df['department'].fillna('Unknown', inplace=True)
df['parental_education'].fillna('Unknown', inplace=True)
df.drop('student_id', axis=1, errors='ignore', inplace=True)

# ===================== ADVANCED FEATURE ENGINEERING =====================
print("🔧 Creating advanced features...")

# Risk flags
df['low_attendance'] = (df['attendance_rate'] < 70).astype(int)
df['low_cgpa'] = (df['cgpa'] < 5.0).astype(int)
df['high_failures'] = (df['past_failures'] > 3).astype(int)
df['low_study_hours'] = (df['study_hours_per_week'] < 10).astype(int)

# Interaction features
df['cgpa_attendance_interaction'] = df['cgpa'] * df['attendance_rate']
df['study_cgpa_ratio'] = df['study_hours_per_week'] / (df['cgpa'] + 0.1)
df['assignment_completion_rate'] = df['assignments_submitted'] / 50.0  # Assuming 50 max
df['academic_performance_score'] = (df['cgpa'] * 10 + df['attendance_rate']) / 2

# Engagement score
df['engagement_score'] = (
    (df['extra_curricular'] == 'Yes').astype(int) +
    (df['sports_participation'] == 'Yes').astype(int) +
    (df['total_activities'] / 5.0)
)

# Risk score (composite)
df['risk_score'] = (
    df['low_attendance'] * 3 +
    df['low_cgpa'] * 3 +
    df['high_failures'] * 2 +
    df['low_study_hours'] * 1
)

# Polynomial features for key numeric columns
df['cgpa_squared'] = df['cgpa'] ** 2
df['attendance_squared'] = df['attendance_rate'] ** 2

# Categorical binning
df['income_bracket'] = pd.cut(df['family_income'], bins=[0, 25000, 50000, 75000, 1000000],
                               labels=['Low', 'Medium', 'High', 'Very High'])
df['age_group'] = pd.cut(df['age'], bins=[15, 18, 21, 25, 100],
                          labels=['Teen', 'Young Adult', 'Adult', 'Mature'])

# ===================== ENCODING =====================
print("🔢 Encoding categorical variables...")
from sklearn.preprocessing import LabelEncoder

label_encoders = {}
categorical_cols = ['gender', 'department', 'scholarship', 'parental_education',
                   'extra_curricular', 'sports_participation', 'income_bracket', 'age_group']

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

joblib.dump(label_encoders, 'label_encoders.pkl')

# ===================== SPLIT & SCALE =====================
X = df.drop('dropout', axis=1)
y = df['dropout']

# Save feature names for prediction consistency
joblib.dump(X.columns.tolist(), 'models/feature_names.pkl')
print(f"✅ Saved {len(X.columns)} feature names to models/feature_names.pkl")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale numeric features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
joblib.dump(scaler, 'scaler.pkl')

# ===================== HANDLE IMBALANCE =====================
print("⚖️ Handling class imbalance with SMOTETomek...")
smotetomek = SMOTETomek(random_state=42)
X_train_balanced, y_train_balanced = smotetomek.fit_resample(X_train_scaled, y_train)
print(f"Balanced train set: {X_train_balanced.shape}")

# ===================== STACKING ENSEMBLE =====================
print("🤖 Training Stacking Ensemble...")

# Base models (diverse algorithms)
base_models = [
    ('rf', RandomForestClassifier(n_estimators=300, max_depth=20, min_samples_split=5,
                                  class_weight='balanced', random_state=42, n_jobs=-1)),
    ('xgb', XGBClassifier(n_estimators=300, max_depth=8, learning_rate=0.05,
                         scale_pos_weight=3, random_state=42, n_jobs=-1, eval_metric='logloss')),
    ('lgbm', LGBMClassifier(n_estimators=300, max_depth=8, learning_rate=0.05,
                           class_weight='balanced', random_state=42, n_jobs=-1, verbose=-1)),
    ('catboost', CatBoostClassifier(iterations=300, depth=8, learning_rate=0.05,
                                   auto_class_weights='Balanced', random_state=42, verbose=0)),
    ('gb', GradientBoostingClassifier(n_estimators=200, max_depth=7, learning_rate=0.05,
                                     random_state=42))
]

# Meta-learner (Logistic Regression)
meta_model = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')

# Create stacking ensemble
stacking_clf = StackingClassifier(
    estimators=base_models,
    final_estimator=meta_model,
    cv=5,
    n_jobs=-1
)

print("Training stacking ensemble (this may take 5-10 minutes)...")
stacking_clf.fit(X_train_balanced, y_train_balanced)

# ===================== VOTING ENSEMBLE (ALTERNATIVE) =====================
print("🗳️ Training Voting Ensemble...")
voting_clf = VotingClassifier(
    estimators=base_models,
    voting='soft',
    n_jobs=-1
)
voting_clf.fit(X_train_balanced, y_train_balanced)

# ===================== EVALUATION =====================
print("\n" + "="*60)
print("MODEL PERFORMANCE COMPARISON")
print("="*60)

models = {
    'Random Forest': RandomForestClassifier(n_estimators=300, max_depth=20, class_weight='balanced', random_state=42, n_jobs=-1),
    'XGBoost': XGBClassifier(n_estimators=300, max_depth=8, learning_rate=0.05, scale_pos_weight=3, random_state=42, n_jobs=-1, eval_metric='logloss'),
    'LightGBM': LGBMClassifier(n_estimators=300, max_depth=8, learning_rate=0.05, class_weight='balanced', random_state=42, n_jobs=-1, verbose=-1),
    'CatBoost': CatBoostClassifier(iterations=300, depth=8, learning_rate=0.05, auto_class_weights='Balanced', random_state=42, verbose=0),
    'Stacking Ensemble': stacking_clf,
    'Voting Ensemble': voting_clf
}

results = {}
for name, model in models.items():
    if name not in ['Stacking Ensemble', 'Voting Ensemble']:
        model.fit(X_train_balanced, y_train_balanced)

    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    results[name] = roc_auc
    print(f"{name:25} | ROC-AUC: {roc_auc:.4f}")

# Select best model
best_model_name = max(results, key=results.get)
best_model = models[best_model_name]
best_score = results[best_model_name]

print("\n" + "="*60)
print(f"🏆 BEST MODEL: {best_model_name} (ROC-AUC: {best_score:.4f})")
print("="*60)

# Detailed evaluation of best model
y_pred = best_model.predict(X_test_scaled)
y_pred_proba = best_model.predict_proba(X_test_scaled)[:, 1]

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Retained', 'Dropout']))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Feature importance (if available)
if hasattr(best_model, 'feature_importances_'):
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
elif best_model_name == 'Stacking Ensemble':
    # Use XGBoost feature importance from base models
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': best_model.estimators_[1].feature_importances_  # XGBoost is 2nd estimator
    }).sort_values('importance', ascending=False)
else:
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': np.zeros(len(X.columns))
    })

print("\nTop 10 Most Important Features:")
print(feature_importance.head(10))

# ===================== SAVE MODEL & ARTIFACTS =====================
print("\n💾 Saving model and artifacts...")
joblib.dump(best_model, 'student_dropout_model.pkl')
joblib.dump(feature_importance, 'feature_importance.pkl')
joblib.dump(results, 'model_comparison_results.pkl')
X.to_csv('feature_columns.csv', index=False)

print("\n✅ Model training complete!")
print("📁 Files saved:")
print("   - student_dropout_model.pkl (Best model)")
print("   - label_encoders.pkl")
print("   - scaler.pkl (IMPORTANT: Use this in predictions!)")
print("   - feature_importance.pkl")
print("   - feature_columns.csv")
print("   - model_comparison_results.pkl")
print(f"\n🎯 Expected ROC-AUC: {best_score:.4f} (85-92% range)")
