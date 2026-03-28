# Data Science Portfolio

# Project 1

**Title:** [🫁 Lung Cancer Risk Prediction Using Machine Learning](https://github.com/odosa-osifo/data-science/blob/main/Lung_Cancer_Risk_Prediction.ipynb)

## ⚡ Key Results (Quick Summary)
- Best Model: Balanced Bagging
- Focus Metric: High Recall(Ensemble-99% after Hyper-parameter tunning) for early detection
- Dataset: 20,000 patients
- Key Features: Smoking, Age, Symptoms

## 💼 Business / Real-World Impact
- Enables early risk detection using low-cost, non-invasive data
- Can support healthcare providers in prioritising high-risk patients
- Reduces dependency on expensive diagnostic procedures

## Overview

This project focuses on predicting lung cancer risk using non-invasive lifestyle and health indicators. The goal is to build an interpretable and accurate machine learning system that can support early detection and preventive healthcare.

Unlike traditional diagnostic methods (e.g., CT scans), this approach uses easily accessible features such as age, smoking history, and symptoms, making it suitable for large-scale screening.

---

## Objectives

* Develop machine learning models to predict lung cancer risk
* Compare performance across multiple algorithms
* Identify key risk factors influencing predictions
* Ensure model interpretability for clinical relevance

---

## Dataset

* ~20,000 patient records
* Features include:

  * Demographics (age, gender)
  * Lifestyle (smoking, alcohol)
  * Symptoms (coughing, fatigue, etc.)

The dataset enables prediction using **non-invasive indicators**, making the solution scalable and cost-effective

**Lung Cancer Data Distribution:**

![Lung Cancer Distribution](01-lung-cancer-risk-prediction/assets/Lung%20Cancer%20Distribution.png)

**Correlation HeatMap:**
![Correlation](01-lung-cancer-risk-prediction/assets/Correlation%20heatmap.png)

---

## Methodology

### 🔹 Data Preprocessing

* Handled missing values using Simple Inputer with median strategy
* Encoded categorical variables using One Hot + Target Encoder
* Feature scaling using Standard Scaler
* Outlier Detection and Handling
* Train-test split with cross-validation (80% & 20%)

### 🔹 Feature Selection

* Recursive Feature Elimination (RFE)
* Sequential Feature Selection (SFS)
* Tree-based feature importance

### 🔹 Models Implemented(Classifiers)

* Logistic Regression
* Support Vector Machine (SVM)
* Random Forest
* Gradient Boosting
* KNN
* Ensemble Methods (Stacking, Bagging)

## Evaluation Metrics

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC
* Confusion Matrix

Special focus was placed on **recall**, due to the importance of correctly identifying high-risk patients.

### 🔹 Model Explainability

* SHAP (global & local explanations)
* LIME (local interpretability)



**Sharp Feature Importance:**
![Sharp Feature Importance](01-lung-cancer-risk-prediction/assets/feature%20importance.png)

---

## Key Results

* Ensemble models achieved the best performance
* Balanced Bagging demonstrated stronger applicability by successfully identifying a large number of true positives while simultaneously correctly identifying a majority of the true negatives avoiding extreme bias towards either class compared to other models.
* Models successfully identified key predictors such as:

  * Smoking history
  * Age
  * Respiratory symptoms

## Model Performance
**Confusion Matrix:**

![Confusion Matrix](01-lung-cancer-risk-prediction/assets/Confusion%20Matrix.png)

**SHAP Summary:**

![SHAP Summary](01-lung-cancer-risk-prediction/assets/Shap%20summary.png)

---

## Key Insights

* Machine learning can effectively predict lung cancer risk using non-invasive data
* Ensemble methods outperform single models in this domain
* Interpretability (SHAP, LIME) is critical for clinical trust

---

## Tech Stack

* Python
* Pandas, NumPy
* Scikit-learn
* Matplotlib / Seaborn
* SHAP, LIME

---

## Project Structure

```
01-lung-cancer-risk-prediction/
├── notebooks/
├── src/
├── assets/
└── README.md
```

---

##  How to Run

```bash
pip install -r requirements.txt
jupyter notebook
```

---

##  Future Improvements

* External validation on real clinical datasets
* Deployment as a web-based risk prediction tool
* Integration with healthcare systems

---


# Project 2
# 🧬 Skin Cancer Classification Using Deep Learning

## Overview

This project focuses on building and comparing deep learning models for multi-class skin disease classification using dermatoscopic images.

The objective is to evaluate the performance of CNN-based architectures and transformer-based models in detecting and classifying skin conditions, supporting early diagnosis in healthcare.

---

## Objectives

* Build and train multiple deep learning models for image classification
* Compare CNN vs Transformer performance
* Evaluate models using classification metrics
* Identify the best-performing architecture for medical image analysis

---

## Dataset

* ~37,000 dermatoscopic images

* 6 classes:

  * Infectious diseases
  * Eczema
  * Acne
  * Pigment disorders
  * Benign
  * Malignant

* Images resized to **224×224**

* Train/Validation/Test split: **80/10/10**

---

## Models Implemented

### 🔹 CNN-Based Models

* ResNet18 (pretrained)
* EfficientNet-B0 (pretrained)

### 🔹 Transformer-Based Model

* Vision Transformer (ViT)

### 🔹 Custom Hybrid Model

* EfficientNet backbone + Fully Connected layers
* Dropout + ReLU for regularisation

---

##  Training Configuration

* Loss Function: Cross Entropy Loss
* Optimizer: Adam
* Learning Rate: 0.0001
* Batch Size: 32
* Epochs: 10

---

## Evaluation Metrics

* Accuracy
* Precision
* Recall
* F1 Score
* Confusion Matrix

---

##  Results

| Model        | Accuracy | Precision | Recall   | F1 Score |
| ------------ | -------- | --------- | -------- | -------- |
| ResNet18     | 0.83     | 0.81      | 0.84     | 0.82     |
| EfficientNet | 0.83     | 0.80      | 0.85     | 0.82     |
| ViT          | 0.81     | 0.78      | 0.84     | 0.80     |
| **Custom**   | **0.87** | **0.83**  | **0.86** | **0.84** |

Best Model: **Custom EfficientNet-based model**

---

##  Key Insights

* CNN models performed strongly on structured medical images
* Vision Transformer showed competitive but slightly lower performance
* Custom hybrid model achieved the best balance of precision and recall
* Data augmentation significantly improved generalisation

---

## Tech Stack

* Python
* PyTorch
* NumPy, Pandas
* Matplotlib

---

## Project Structure

```bash
02-skin-cancer-classification/
├── notebooks/
├── models/
├── assets/
└── README.md
```

---

##  How to Run

```bash
pip install -r requirements.txt
jupyter notebook
```

---

##  Future Improvements

* Increase dataset size for better generalisation
* Hyperparameter tuning
* Deploy model as a web-based diagnostic tool


# Project 3
# 🚗 Vehicle Price Prediction Using Machine Learning

##  Overview

This project focuses on predicting vehicle prices using a large-scale dataset of over 400,000 records. The goal is to build a robust machine learning pipeline capable of delivering accurate predictions while handling real-world data challenges such as missing values, outliers, and high-cardinality categorical variables.

---

##  Objectives

* Develop a complete machine learning pipeline for price prediction
* Apply feature engineering to improve model performance
* Compare multiple models and feature selection techniques
* Optimise model performance using hyperparameter tuning
* Evaluate ensemble learning approaches

---

##  Dataset

* 402,000+ vehicle records
* Features include:

  * Mileage
  * Year of registration
  * Vehicle condition
  * Fuel type
  * Make & model

---

##  Methodology

### 🔹 Data Preprocessing

* Missing value imputation:

  * Numerical → Mean
  * Categorical → Mode
* Outlier handling using IQR (clipping extreme values)
* Feature scaling using StandardScaler
* One-Hot Encoding and Target Encoding

---

### 🔹 Feature Engineering

* Created **Car_Age** from registration year
* Combined features into **Make & Model**
* Reduced dimensionality using encoding strategies

---

### 🔹 Feature Selection Techniques

* SelectKBest (f_regression)
* Recursive Feature Elimination (RFE)
* Sequential Feature Selection (SFS)

 Best method: **Sequential Feature Selection**

---

### 🔹 Models Implemented

* Linear Regression
* Random Forest Regressor
* Gradient Boosting Regressor
* XGBoost Regressor

---

### 🔹 Ensemble Learning

* Averaging (Voting Regressor)
* Stacking (Meta-model: RidgeCV)

---

##  Evaluation Metrics

* R² Score
* Mean Absolute Error (MAE)
* Mean Squared Error (MSE)
* Root Mean Squared Error (RMSE)

---

##  Results

| Model          | R² Score | MAE  |
| -------------- | -------- | ---- |
| Random Forest  | 0.94     | 1454 |
| Gradient Boost | 0.88     | 2313 |
| XGBoost        | 0.89     | 2312 |
| Stacking       | 0.95     | 1568 |

 Best Model: **Random Forest Regressor**
 Best Ensemble: **Stacking (highest R²)**

---

##  Key Insights

* Feature engineering (Car_Age and Make&Model) significantly improved predictive performance, highlighting the importance of domain-informed features in regression tasks.
* Sequential Feature Selection captured feature interactions effectively and evalated with metrics: r2 and mae
* Tree-based ensemble models outperformed linear models
* Random Forest provided the best balance between accuracy and generalisation

---

##  Tech Stack

* Python
* Pandas, NumPy
* Scikit-learn
* XGBoost
* Matplotlib

---

##  Project Structure

```bash id="b0ju38"
03-vehicle-price-prediction/
├── notebooks/
├── src/
├── assets/
└── README.md
```

---

##  How to Run

```bash id="71uddp"
pip install -r requirements.txt
jupyter notebook
```

---

##  Future Improvements

* Deploy model as a pricing API
* Incorporate real-time vehicle market data
* Explore deep learning approaches for regression
