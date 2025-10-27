# 🎓 Student Disengagement Prediction System

## 🚀 Quick Start

1. **Run the Application**
   ```
   streamlit run app.py
   ```

2. **Access the Dashboard**
   - Open browser: `http://localhost:8501`

## 📊 Real Dataset Stats

- **Total Students**: 20,000
- **Dropout Students**: 5,672 (28.4%)
- **Features**: 21 engineered features
- **Model**: Random Forest Classifier
- **Accuracy**: 70.87%
- **F1 Score**: 0.4538
- **ROC-AUC**: 0.7069

## 🎯 Key Features

### 1. **Dashboard** 📈
- Real-time KPIs (Total students, at-risk count, avg attendance, avg CGPA)
- Interactive visualizations (attendance distribution, department analysis, trend charts)
- Beautiful gradient UI with glassmorphism effects

### 2. **Students** 👥
- Searchable student database
- Filter by department, age, risk status
- Individual student risk profiles
- Detailed metrics per student

### 3. **Prediction** 🎯
- Interactive risk simulator
- Adjust academic factors (attendance, CGPA, assignments, failures)
- Adjust personal factors (study hours, projects, sports, scholarship)
- Real-time risk score calculation

### 4. **Analytics** 📊
- Feature importance analysis
- ROC curve visualization
- Confusion matrix heatmap
- Model performance metrics

### 5. **AI Assistant** 🤖
- Interactive chatbot
- Ask questions about students at risk
- Get intervention recommendations
- Query statistics

## 🎨 UI Highlights

- **Gradient Background**: Purple (#667eea) to dark purple (#764ba2)
- **Glassmorphism Cards**: Frosted glass effect with subtle shadows
- **Smooth Animations**: Fade-in effects, hover transitions
- **Responsive Design**: Works on all screen sizes
- **Modern Icons**: Emojis for intuitive navigation

## 📁 Project Structure

```
omnitrics/
├── app.py                    # Main Streamlit dashboard (800+ lines)
├── config.py                 # Configuration & constants
├── preprocess.py             # Data cleaning pipeline
├── train_model.py            # ML training pipeline
├── requirements.txt          # Python dependencies
├── data/
│   ├── raw/                  # Original dataset
│   └── processed/            # Cleaned data
│       └── cleaned_student_data.csv
├── models/
│   ├── student_model.pkl     # Trained Random Forest model
│   ├── scaler.pkl           # Feature scaler
│   ├── label_encoders.pkl   # Categorical encoders
│   └── feature_names.pkl    # Feature list
└── reports/
    └── figures/
        ├── feature_importance.png
        ├── roc_curve.png
        └── confusion_matrix.png
```

## 🔑 Key Technologies

- **ML**: XGBoost, Random Forest, Gradient Boosting, SMOTE
- **UI**: Streamlit, Custom CSS, Plotly
- **Data**: Pandas, NumPy, Scikit-learn
- **Viz**: Matplotlib, Seaborn, Plotly

## 📈 Model Performance

### Top Features by Importance:
1. **Past Failures** (23.7%)
2. **Attendance Rate** (12.0%)
3. **CGPA** (10.4%)
4. **Academic Score** (7.1%)
5. **Study Hours/Week** (5.6%)

### Classification Metrics:
- **True Negatives**: 2,351 (correctly identified safe students)
- **True Positives**: 484 (correctly identified at-risk students)
- **False Positives**: 515 (false alarms)
- **False Negatives**: 650 (missed at-risk students)

## 🎯 Intervention Strategies

### High Risk (>60%)
- Immediate academic counseling
- Weekly mentorship sessions
- Personalized study plans
- Financial aid review
- Mental health support

### Medium Risk (30-60%)
- Bi-weekly check-ins
- Peer tutoring programs
- Study groups formation
- Time management workshops
- Career guidance

### Low Risk (<30%)
- Monthly monitoring
- Academic excellence programs
- Leadership opportunities
- Scholarship recommendations
- Peer mentoring roles

## 💡 Hackathon Presentation Tips

1. **Start with the Problem**: 28.4% dropout rate in higher education
2. **Show the Dashboard**: Live demo of beautiful UI
3. **Highlight Real Data**: 20,000 actual student records
4. **Explain ML**: Random Forest with 70.87% accuracy
5. **Demonstrate Impact**: Early intervention can save students
6. **Show Features**: All 5 pages (Dashboard, Students, Prediction, Analytics, AI Assistant)
7. **Emphasize Design**: Modern UI with gradient effects and smooth animations

## 🏆 Competitive Advantages

✅ **Real Dataset**: Not synthetic, actual 20,000+ student records
✅ **Beautiful UI**: Professional gradient design with glassmorphism
✅ **Multiple Models**: Tested RF, GB, XGBoost - chose best performer
✅ **Comprehensive**: 5 full pages with different functionalities
✅ **Actionable**: Specific intervention strategies for each risk level
✅ **Scalable**: Easy to add more features or retrain model
✅ **Interactive**: Prediction simulator for what-if scenarios

## 📞 Demo Flow

1. **Open Dashboard** → Show KPIs and charts
2. **Navigate to Students** → Filter and search functionality
3. **Go to Prediction** → Interactive simulator
4. **Show Analytics** → Feature importance and model performance
5. **Try AI Assistant** → Ask sample questions
6. **Highlight Impact** → Early intervention saves students

---

**Built with ❤️ for Education**
