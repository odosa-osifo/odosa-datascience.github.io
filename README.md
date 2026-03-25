# data-science-projects
# 🫁 Lung Cancer Risk Prediction Using Machine Learning

## 📌 Overview

This project focuses on predicting lung cancer risk using non-invasive lifestyle and health indicators. The goal is to build an interpretable and accurate machine learning system that can support early detection and preventive healthcare.

Unlike traditional diagnostic methods (e.g., CT scans), this approach uses easily accessible features such as age, smoking history, and symptoms, making it suitable for large-scale screening.

---

## 🎯 Objectives

* Develop machine learning models to predict lung cancer risk
* Compare performance across multiple algorithms
* Identify key risk factors influencing predictions
* Ensure model interpretability for clinical relevance

---

## 📊 Dataset

* ~20,000 patient records
* Features include:

  * Demographics (age, gender)
  * Lifestyle (smoking, alcohol)
  * Symptoms (coughing, fatigue, etc.)

The dataset enables prediction using **non-invasive indicators**, making the solution scalable and cost-effective.

---

## 🧠 Methodology

### 🔹 Data Preprocessing

* Handling missing values
* Encoding categorical variables
* Feature scaling
* Train-test split with cross-validation

### 🔹 Models Implemented

* Logistic Regression
* Support Vector Machine (SVM)
* Random Forest
* Gradient Boosting
* Ensemble Methods (Stacking, Bagging)

### 🔹 Feature Selection

* Recursive Feature Elimination (RFE)
* Tree-based feature importance

### 🔹 Model Explainability

* SHAP (global & local explanations)
* LIME (local interpretability)

---

## 📈 Evaluation Metrics

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC
* Confusion Matrix

Special focus was placed on **recall**, due to the importance of correctly identifying high-risk patients.

---

## 🏆 Key Results

* Ensemble models achieved the best performance
* Balanced Bagging demonstrated strong real-world applicability
* Models successfully identified key predictors such as:

  * Smoking history
  * Age
  * Respiratory symptoms

---

## 💡 Key Insights

* Machine learning can effectively predict lung cancer risk using non-invasive data
* Ensemble methods outperform single models in this domain
* Interpretability (SHAP, LIME) is critical for clinical trust

---

## ⚙️ Tech Stack

* Python
* Pandas, NumPy
* Scikit-learn
* Matplotlib / Seaborn
* SHAP, LIME

---

## 📂 Project Structure

```
01-lung-cancer-risk-prediction/
├── notebooks/
├── src/
├── assets/
└── README.md
```

---

## 🚀 How to Run

```bash
pip install -r requirements.txt
jupyter notebook
```

---

## 📌 Future Improvements

* External validation on real clinical datasets
* Deployment as a web-based risk prediction tool
* Integration with healthcare systems

---

## 🔗 Full Report

This project is based on my MSc dissertation:

📄 Lung Cancer Risk Prediction Dissertation
