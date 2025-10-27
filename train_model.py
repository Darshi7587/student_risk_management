"""Train advanced ML model with real student data"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score, 
                            classification_report, confusion_matrix, roc_curve)
from imblearn.over_sampling import SMOTE
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from config import *

print("🤖 Starting Advanced Model Training...")
print("="*60)

# Load preprocessed data
df = pd.read_csv(CLEANED_DATA_FILE)
print(f"✓ Loaded {len(df):,} student records")

# Prepare features
exclude_cols = ['student_id', 'dropout', 'disengaged', 'gender', 'department', 
                'scholarship', 'parental_education', 'extra_curricular', 
                'sports_participation']

feature_cols = [col for col in df.columns if col not in exclude_cols]
X = df[feature_cols].fillna(0)
y = df['disengaged']

print(f"✓ Features: {len(feature_cols)}")
print(f"✓ Target distribution: {y.value_counts().to_dict()}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"✓ Split: Train={len(X_train):,}, Test={len(X_test):,}")

# Handle class imbalance with SMOTE
print("\n⚖️ Balancing classes with SMOTE...")
smote = SMOTE(random_state=42, k_neighbors=3)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
print(f"✓ After SMOTE: {np.bincount(y_train_balanced)}")

# Train multiple models
print("\n🎯 Training Multiple Models...")
print("-"*60)

models = {
    'Random Forest': RandomForestClassifier(
        n_estimators=200, max_depth=15, min_samples_split=5,
        random_state=42, n_jobs=-1
    ),
    'Gradient Boosting': GradientBoostingClassifier(
        n_estimators=150, max_depth=10, learning_rate=0.1,
        random_state=42
    ),
    'XGBoost': XGBClassifier(
        n_estimators=200, max_depth=12, learning_rate=0.1,
        random_state=42, use_label_encoder=False, eval_metric='logloss'
    )
}

results = {}
best_model = None
best_score = 0

for name, model in models.items():
    print(f"\nTraining {name}...")
    
    # Train
    model.fit(X_train_balanced, y_train_balanced)
    
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    results[name] = {
        'model': model,
        'accuracy': accuracy,
        'f1_score': f1,
        'roc_auc': roc_auc
    }
    
    print(f"  ✓ Accuracy:  {accuracy:.4f}")
    print(f"  ✓ F1 Score:  {f1:.4f}")
    print(f"  ✓ ROC-AUC:   {roc_auc:.4f}")
    
    # Track best model
    if f1 > best_score:
        best_score = f1
        best_model = model
        best_name = name

print(f"\n{'='*60}")
print(f"🏆 Best Model: {best_name} (F1: {best_score:.4f})")
print(f"{'='*60}")

# Final evaluation with best model
print(f"\n📊 Final Model Evaluation:")
print("-"*60)

y_pred_final = best_model.predict(X_test)
y_pred_proba_final = best_model.predict_proba(X_test)[:, 1]

print("\nClassification Report:")
print(classification_report(y_test, y_pred_final, 
                          target_names=['Safe', 'At Risk'],
                          digits=4))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_final)
print("\nConfusion Matrix:")
print(cm)
print(f"True Negatives:  {cm[0][0]}")
print(f"False Positives: {cm[0][1]}")
print(f"False Negatives: {cm[1][0]}")
print(f"True Positives:  {cm[1][1]}")

# Save best model
joblib.dump(best_model, MODEL_FILE)
print(f"\n✓ Model saved to: {MODEL_FILE}")

# Feature Importance
print(f"\n📈 Generating Feature Importance Analysis...")
if hasattr(best_model, 'feature_importances_'):
    importances = best_model.feature_importances_
    indices = np.argsort(importances)[::-1][:20]
    
    feature_importance_df = pd.DataFrame({
        'Feature': [feature_cols[i] for i in indices],
        'Importance': importances[indices]
    })
    
    print("\nTop 10 Most Important Features:")
    print(feature_importance_df.head(10).to_string(index=False))
    
    # Plot
    plt.figure(figsize=(12, 8))
    plt.barh(range(len(indices)), importances[indices], color='#667eea')
    plt.yticks(range(len(indices)), [feature_cols[i] for i in indices])
    plt.xlabel('Feature Importance', fontsize=12, fontweight='bold')
    plt.title('Top 20 Feature Importances for Dropout Prediction', 
             fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'feature_importance.png', dpi=300, bbox_inches='tight')
    print(f"✓ Feature importance plot saved")

# ROC Curve
plt.figure(figsize=(10, 8))
fpr, tpr, _ = roc_curve(y_test, y_pred_proba_final)
roc_auc_final = roc_auc_score(y_test, y_pred_proba_final)

plt.plot(fpr, tpr, color='#667eea', lw=3, 
        label=f'ROC Curve (AUC = {roc_auc_final:.4f})')
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', label='Random')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
plt.ylabel('True Positive Rate', fontsize=12, fontweight='bold')
plt.title('ROC Curve - Student Dropout Prediction', fontsize=14, fontweight='bold')
plt.legend(loc="lower right", fontsize=11)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'roc_curve.png', dpi=300, bbox_inches='tight')
print(f"✓ ROC curve saved")

# Confusion Matrix Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
           xticklabels=['Safe', 'At Risk'],
           yticklabels=['Safe', 'At Risk'],
           annot_kws={'size': 16, 'weight': 'bold'})
plt.xlabel('Predicted', fontsize=12, fontweight='bold')
plt.ylabel('Actual', fontsize=12, fontweight='bold')
plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
print(f"✓ Confusion matrix saved")

print(f"\n{'='*60}")
print(f"✅ TRAINING COMPLETE!")
print(f"{'='*60}")
print(f"Model Type:     {best_name}")
print(f"Accuracy:       {accuracy_score(y_test, y_pred_final):.4f}")
print(f"F1 Score:       {f1_score(y_test, y_pred_final):.4f}")
print(f"ROC-AUC:        {roc_auc_score(y_test, y_pred_proba_final):.4f}")
print(f"Model saved:    {MODEL_FILE}")
print(f"{'='*60}\n")
