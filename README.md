# 🎓 Smart Student Performance Prediction System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Machine Learning](https://img.shields.io/badge/ML-Scikit--learn-orange.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-Latest-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

**A comprehensive machine learning system that predicts student academic performance using multiple algorithms and advanced data preprocessing techniques.**

[Features](#-features) • [Dataset](#-dataset) • [Methodology](#-methodology) • [Results](#-results) • [Installation](#-installation) • [Usage](#-usage)

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Team Members](#-team-members)
- [Features](#-features)
- [Dataset](#-dataset)
- [Methodology](#-methodology)
- [Results](#-results)
- [Installation](#-installation)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Technologies Used](#-technologies-used)
- [Key Insights](#-key-insights)
- [Future Improvements](#-future-improvements)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🎯 Overview

This project implements a **Smart Student Performance Prediction System** that leverages machine learning algorithms to predict student academic outcomes based on various factors including study habits, attendance, previous academic performance, and personal characteristics. The system addresses three distinct prediction tasks:

1. **Binary Classification**: Predict Pass/Fail status
2. **Multi-class Classification**: Predict Final Grade (A, B, C, D, F)
3. **Regression**: Predict Final Score (continuous value)

The project demonstrates expertise in data preprocessing, feature engineering, hyperparameter tuning, and handling class imbalance using advanced techniques like SMOTE.

---

## 👥 Team Members

| Name | Student ID |
|------|-----------|
| Ahmed Nasr | 202200294 |
| Ahmed Salah | 202200212 |
| Ahmed Fares | 202200977 |
| Hassan Ahmed | 202202121 |

---

## ✨ Features

- **Comprehensive Data Preprocessing**
  - Missing value imputation using Iterative Imputer and Simple Imputer
  - Categorical encoding with One-Hot Encoding
  - Outlier detection and capping using IQR method
  - Feature scaling with StandardScaler

- **Multiple ML Models**
  - Logistic Regression
  - Support Vector Machines (Linear & RBF kernels)
  - K-Nearest Neighbors
  - Random Forest
  - XGBoost
  - Linear Regression, Ridge, Lasso

- **Advanced Techniques**
  - SMOTE for handling class imbalance
  - Hyperparameter tuning using GridSearchCV and RandomizedSearchCV
  - Cross-validation for robust model evaluation
  - Feature importance analysis

- **Comprehensive Evaluation**
  - Multiple metrics (Accuracy, F1-Score, ROC-AUC, R², RMSE, MAE)
  - Confusion matrices
  - ROC curves
  - Residual plots
  - Classification reports

---

## 📊 Dataset

### Dataset Information
- **Total Records**: 20,000 students
- **Features**: 41 attributes
- **Target Variables**: 
  - `final_score` (Regression)
  - `final_grade` (Multi-class: A, B, C, D, F)
  - `pass_fail` (Binary: Pass/Fail)

### Key Features Include:
- **Demographic**: Age, Gender, Parent Income, Number of Siblings
- **Academic History**: Previous GPA, High School Grade, Failed Courses
- **Background Scores**: Math, Language, Science
- **Study Habits**: Study Hours, Study Time per Week, Library Visits
- **Engagement**: Attendance Rates, Assignment Submission, Quiz Scores
- **Behavioral**: Stress Level, Sleep Hours, Motivation Level, Exam Anxiety
- **Course Context**: Course Type, Class Size, Teacher Experience, Prerequisites

### Data Quality
- Missing values handled with advanced imputation techniques
- Outliers detected and capped at 1st and 99th percentiles
- Class imbalance addressed using SMOTE

---

## 🔬 Methodology

### 1. Exploratory Data Analysis (EDA)
- Distribution analysis of target variables
- Correlation matrix visualization
- Missing value analysis
- Class distribution examination

### 2. Data Preprocessing Pipeline

#### Missing Value Handling
- **Categorical**: Most frequent imputation
- **Numerical**: Iterative Imputer (models each feature as a function of other features)
- Rows with missing target variables are dropped

#### Feature Engineering
- One-Hot Encoding for categorical variables (Gender, Part-time Job, Course Type)
- Outlier capping using IQR method (3×IQR threshold)
- Standard Scaling for all numerical features

### 3. Model Training Strategy

#### Binary Classification (Pass/Fail)
- **Challenge**: Severe class imbalance (90% Pass, 7% Fail)
- **Solution**: SMOTE oversampling applied to training data
- **Models Tested**: Logistic Regression, SVM (Linear & RBF), KNN, Random Forest, XGBoost
- **Optimization**: RandomizedSearchCV with F1-score as metric
- **Best Model**: XGBoost

#### Regression (Final Score)
- **Models Tested**: Linear Regression, Ridge, Lasso, Random Forest, XGBoost
- **Optimization**: RandomizedSearchCV with negative MSE as metric
- **Best Model**: Lasso Regression
- **Performance**: R² = 0.77, RMSE = 5.29

#### Multi-class Classification (Final Grade)
- **Challenge**: Extreme class imbalance (70% Grade C)
- **Approaches Tested**:
  1. XGBoost with SMOTE
  2. One-vs-Rest Random Forest with balanced class weights
- **Optimization**: GridSearchCV with Macro F1-score
- **Best Model**: One-vs-Rest Random Forest

### 4. Model Evaluation
- Train/Test Split: 80/20
- Stratified sampling for classification tasks
- Comprehensive metrics for each task type
- Visualizations: Confusion matrices, ROC curves, residual plots

---

## 📈 Results

### Binary Classification (Pass/Fail)
- **Best Model**: XGBoost
- **Accuracy**: 95%
- **F1-Score**: 0.94 (weighted), 0.76 (macro)
- **ROC-AUC**: High performance on minority class

### Regression (Final Score)
- **Best Model**: Lasso Regression
- **R² Score**: 0.7727
- **RMSE**: 5.29
- **MAE**: Low error rate indicating good predictive capability

### Multi-class Classification (Final Grade)
- **Best Model**: One-vs-Rest Random Forest
- **Accuracy**: 69%
- **Macro F1-Score**: 0.22
- **Weighted F1-Score**: 0.59
- **Note**: Performance reflects the challenge of extreme class imbalance

### Key Findings
- **Top Predictive Features**: 
  - Previous academic performance (GPA, high school grades)
  - Engagement metrics (attendance, assignment submission)
  - Study habits (study hours, library visits)
  - Background scores (math, language, science)

---

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Step 1: Clone the Repository
```bash
git clone https://github.com/yourusername/student-performance-prediction.git
cd student-performance-prediction
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

If `requirements.txt` doesn't exist, install packages manually:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost imbalanced-learn jupyter
```

### Step 3: Download Dataset
Ensure `Term_Project_Dataset_20K.csv` is in the project root directory.

---

## 💻 Usage

### Running the Notebook

1. **Start Jupyter Notebook**:
```bash
jupyter notebook
```

2. **Open the Project Notebook**:
   - Navigate to `full_report.ipynb`
   - Run all cells sequentially

### Key Sections in the Notebook:

1. **Data Loading**: Load and inspect the dataset
2. **EDA**: Exploratory data analysis and visualizations
3. **Preprocessing**: Data cleaning and feature engineering
4. **Model Training**: Train and tune multiple models
5. **Evaluation**: Comprehensive model evaluation and visualization

### Expected Runtime
- Full pipeline execution: ~40-60 minutes (depending on hardware)
- Individual model training: 5-15 minutes per model

---

## 📁 Project Structure

```
student-performance-prediction/
│
├── full_report.ipynb              # Main Jupyter notebook with complete analysis
├── Term_Project_Dataset_20K.csv    # Dataset (20,000 records)
├── Machine Learning Term Project.pdf  # Project requirements document
├── README.md                       # This file
└── requirements.txt                # Python dependencies (to be created)
```

---

## 🛠️ Technologies Used

### Core Libraries
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **matplotlib**: Data visualization
- **seaborn**: Statistical data visualization

### Machine Learning
- **scikit-learn**: 
  - Models: Logistic Regression, SVM, KNN, Random Forest, Linear Regression, Ridge, Lasso
  - Preprocessing: StandardScaler, IterativeImputer, SimpleImputer
  - Model Selection: GridSearchCV, RandomizedSearchCV, cross_val_score
  - Metrics: Accuracy, F1-Score, ROC-AUC, R², RMSE, MAE

- **XGBoost**: Gradient boosting framework
- **imbalanced-learn**: SMOTE for handling class imbalance

### Development Environment
- **Jupyter Notebook**: Interactive development and analysis

---

## 🔍 Key Insights

### Data Insights
1. **Class Imbalance**: Significant imbalance in both binary and multi-class targets
2. **Missing Values**: ~3% missing values across various features
3. **Feature Correlations**: Strong correlations between academic performance indicators

### Model Insights
1. **XGBoost** excels in binary classification tasks with imbalanced data
2. **Lasso Regression** provides good regularization for regression tasks
3. **One-vs-Rest** strategy helps with multi-class imbalance
4. **SMOTE** effectively balances training data for better minority class performance

### Performance Insights
1. Regression task shows strong predictive capability (R² = 0.77)
2. Binary classification achieves high accuracy (95%)
3. Multi-class classification remains challenging due to extreme imbalance
4. Feature importance analysis reveals academic history as strongest predictor

---

## 🔮 Future Improvements

### Model Enhancements
- [ ] Experiment with deep learning models (Neural Networks)
- [ ] Implement ensemble methods combining multiple models
- [ ] Try advanced techniques like CatBoost or LightGBM
- [ ] Apply feature selection techniques to reduce dimensionality

### Data Improvements
- [ ] Collect more data for minority classes
- [ ] Feature engineering: Create interaction features
- [ ] Time-series analysis if temporal data available
- [ ] External data integration (e.g., socioeconomic indicators)

### Technical Improvements
- [ ] Create a web application for predictions
- [ ] Implement model versioning and MLOps practices
- [ ] Add automated hyperparameter optimization (Optuna, Hyperopt)
- [ ] Create API endpoints for model inference
- [ ] Implement model explainability (SHAP values)

### Documentation
- [ ] Add detailed code comments
- [ ] Create tutorial notebooks for each model
- [ ] Document hyperparameter tuning process
- [ ] Add performance benchmarks

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Contribution Guidelines
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 📧 Contact

For questions, suggestions, or collaborations, please contact:

- **Ahmed Nasr** - [GitHub](https://github.com/ahmednasr)
- **Ahmed Salah** - [GitHub](https://github.com/ahmedsalah)
- **Ahmed Fares** - [GitHub](https://github.com/ahmedfares)
- **Hassan Ahmed** - [GitHub](https://github.com/hassanahmed)

---

## 🙏 Acknowledgments

- Special thanks to the course instructors for providing the dataset and project requirements
- Scikit-learn and XGBoost communities for excellent documentation
- Open-source ML community for tools and resources

---

<div align="center">

**Made with ❤️ by the Student Performance Prediction Team**

⭐ Star this repo if you find it helpful!

</div>

