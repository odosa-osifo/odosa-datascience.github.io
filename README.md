# Data Science Portfolio

# Project 1

**Title:** [🫁 Lung Cancer Risk Prediction Using Machine Learning](https://colab.research.google.com/drive/1dBYuzVBkKIbPC3_p1D85HeKoOXOZtApZ)

![Overview](Lung-Cancer-Illustration-Feature.png)

This project demonstrates how machine learning can be applied to real-world healthcare challenges, balancing predictive performance with model interpretability to support clinical decision-making.

## ⚡ Key Results (Quick Summary)
- Best Model: Balanced Bagging Classifier
- Focus Metric: High Recall (99%) to prioritise early detection of high-risk patients
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

* Handled missing values using Simple Imputer with median strategy
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

In healthcare applications, recall is prioritised to minimise false negatives, ensuring high-risk patients are not missed.

### 🔹 Model Explainability

* SHAP (global & local explanations)
* LIME (local interpretability)


**Local Explanations:**
![Local Explanations](01-lung-cancer-risk-prediction/assets/Local%20Explanations.png)

**SHAP Feature Importance:**
![SHAP Feature Importance](01-lung-cancer-risk-prediction/assets/feature%20importance.png)

---

## Key Results
###  Model Comparison

| Model               |Accuracy | Recall | F1 Score |
|---------------------|---------|--------|----------|
| Logistic Regression | 0.50     | 0.51     | 0.64|
| Random Forest       | 0.85     | 0.96     | 0.91|
| Gradient Boosting   | 0.68     | 0.74     | 0.80|
| KNN                 | 0.74     | 0.82     | 0.84|
| SVM                 | 0.82     | 0.93     | 0.90|
| Balanced Bagging    | **0.70**  | **99%**| **0.81**|



* Balanced Bagging provided the best balance between recall and precision, making it more suitable for real-world clinical applications where both false positives and false negatives matter.
* Ensemble models achieved the best performance
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
├── assets/
└── README.md
```

---

##  How to Run

1. Clone the repository:

```
git clone https://github.com/odosa-osifo/data-science.git
cd 01-lung-cancer-risk-prediction
```
2. Install Dependencies
```
pip install -r requirements.txt
```
3. Run Notebook
```
jupyter notebook notebooks/lung_cancer_model.ipynb
```
---

##  Future Improvements

* External validation on real clinical datasets
* Deployment as a web-based risk prediction tool
* Integration with healthcare systems

##  Skills Demonstrated
- End-to-end machine learning pipeline development
- Feature engineering and selection
- Ensemble learning techniques
- Model interpretability (SHAP, LIME)
- Handling real-world healthcare datasets

---


# Project 2
**Title: 🧬 Skin Cancer Classification Using Deep Learning**
Automated multi-class classification of skin diseases using CNN and Transformer-based deep learning models.

**Models**

[VIT](https://colab.research.google.com/drive/1hsPcLgQ2Toi7oXAe3IAfhy_L37ddR3GI)
[EffientNet](https://colab.research.google.com/drive/1Xhr9xQBtjzFW4M_wMF7RVcYNEztothPh)
[Resnet18](https://colab.research.google.com/drive/198J0s4VQucHTDFk4W2FTrR1vOkDxryF_)
[Custom_Built](https://colab.research.google.com/drive/1Qy_6YZdq-O_kyiDfx2pkdZEPrzeVoU9_)

## ⚡ Key Results (Quick Summary)

- Best Model: Custom EfficientNet-based model
- Accuracy: 87%
- Dataset: 37,000 images
- Task: Multi-class classification (6 classes)
- Approach: CNN vs Transformer comparison

## 💼 Business / Real-World Impact

- Supports early detection of skin diseases, including potential malignancies
- Can assist dermatologists in diagnosis and reduce manual workload
- Enables scalable AI-driven screening using medical imaging

## Overview

This project focuses on building and comparing deep learning models for multi-class skin disease classification using dermatoscopic images.

The objective is to evaluate the performance of CNN-based architectures and transformer-based models in detecting and classifying skin conditions, supporting early diagnosis in healthcare.

This project demonstrates the application of both convolutional neural networks (CNNs) and transformer-based architectures in medical image classification.

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


# Project 4

**Title:** SQL Data Definition and Manipulation Language - Sales & Workplace Safety Data

**SQL Code:** [Sales & Workplace Safety Data](https://github.com/odosa-osifo/odosa-osifo.github.io/blob/main/Sales_Data.sql)

**SQL Skills Used:** 
This project demonstrates the use of SQL to retrieve, transform, and analyse structured data from relational databases. The work covers employee data analysis and workplace safety incident analysis, using SQL queries to answer business questions, identify patterns, perform aggregation and generate actionable insights. 

The project includes the following SQL operations:

- Data Retrieval (SELECT)
Extracted specific columns from the database, such as employee names and department information, to retrieve relevant records for analysis.

- Text Transformation (UPPER, LEFT)
Applied string functions to manipulate textual data, including converting employee names to uppercase and extracting the first four characters of employee names for formatting and analysis purposes.

- Data Aggregation (COUNT)
Calculated the number of employees within specific departments (e.g., HR) to generate summary insights about workforce distribution.

- Date Functions (GETDATE)
Retrieved the current system date using SQL date functions to demonstrate dynamic date handling within queries.

- SQL Joins (LEFT, INNER)
Joined data from multiple tables using unique identifiers to link tables and performed filtering and analysis

- String Parsing (CHARINDEX)
Extracted location names from the address column by identifying and isolating text that appears before brackets, enabling cleaner geographic data representation.

- Database Structure Management (CREATE TABLE)
Created a new table by defining its structure to support additional data storage and database organization.

- Data Filtering (WHERE)
Applied conditional filtering to retrieve only relevant records, such as employees belonging to a specific department.

- Data Aggregation (COUNT, SUM, AVG)
Used aggregate functions to calculate incident counts, total incident costs, average costs, and average days lost across different categories such as plant, department, injury location, and incident type.

- Grouping and Categorisation (GROUP BY)
Grouped records by plant, department, report type, age group, gender, incident type, shift, month, and year to identify patterns and compare safety performance across multiple dimensions.

- Sorting and Ranking (ORDER BY, TOP)
Applied sorting and ranking techniques to identify the highest-risk plants, departments, age groups, and incident types based on frequency, cost, and severity.

**Project Description:** This portfolio project showcases practical SQL applications across multiple datasets, including employee records and workplace safety incident data. The objective was to demonstrate how SQL can be used to manage, query, and analyse structured business data efficiently in a relational database. 

The queries performed in this project include retrieving employee and workplace safety information, transforming text data, calculating department-level summaries, extracting specific elements from string fields, and generating system date values. These operations demonstrate practical applications of SQL functions for handling real-world business data.

Across the queries, SQL was used to extract relevant records, summarise trends, aggregate, filter and analyse operational performance, and support decision-making through clear and targeted reporting. Overall, the project highlights foundational SQL skills essential for data analysis, reporting, and database management.

Additionally, the project includes database structure management by creating new tables to support data organization and storage.

Overall, this project showcases fundamental SQL skills including data retrieval, string manipulation, aggregation, filtering, and database management, which are essential for data analysts and data scientists working with relational databases.

**Technology Used:** SQL server

**Recommendations:**
- Maintain clean, well-structured, and standardised datasets to improve the accuracy and efficiency of SQL analysis.
- Use SQL-based reporting to monitor employee data and workplace safety metrics more consistently over time.
- Focus safety improvement efforts on high-risk plants, departments, shifts, and incident types identified in the analysis.
- Implement regular automated reports to support faster decision-making and operational monitoring.

