# Data-Science-Projects
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



# 🧬 Skin Cancer Classification Using Deep Learning

## 📌 Overview

This project focuses on building and comparing deep learning models for multi-class skin disease classification using dermatoscopic images.

The objective is to evaluate the performance of CNN-based architectures and transformer-based models in detecting and classifying skin conditions, supporting early diagnosis in healthcare.

---

## 🎯 Objectives

* Build and train multiple deep learning models for image classification
* Compare CNN vs Transformer performance
* Evaluate models using classification metrics
* Identify the best-performing architecture for medical image analysis

---

## 📊 Dataset

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

## 🧠 Models Implemented

### 🔹 CNN-Based Models

* ResNet18 (pretrained)
* EfficientNet-B0 (pretrained)

### 🔹 Transformer-Based Model

* Vision Transformer (ViT)

### 🔹 Custom Hybrid Model

* EfficientNet backbone + Fully Connected layers
* Dropout + ReLU for regularisation

---

## ⚙️ Training Configuration

* Loss Function: Cross Entropy Loss
* Optimizer: Adam
* Learning Rate: 0.0001
* Batch Size: 32
* Epochs: 10

---

## 📈 Evaluation Metrics

* Accuracy
* Precision
* Recall
* F1 Score
* Confusion Matrix

---

## 🏆 Results

| Model        | Accuracy | Precision | Recall   | F1 Score |
| ------------ | -------- | --------- | -------- | -------- |
| ResNet18     | 0.83     | 0.81      | 0.84     | 0.82     |
| EfficientNet | 0.83     | 0.80      | 0.85     | 0.82     |
| ViT          | 0.81     | 0.78      | 0.84     | 0.80     |
| **Custom**   | **0.87** | **0.83**  | **0.86** | **0.84** |

👉 Best Model: **Custom EfficientNet-based model**

---

## 💡 Key Insights

* CNN models performed strongly on structured medical images
* Vision Transformer showed competitive but slightly lower performance
* Custom hybrid model achieved the best balance of precision and recall
* Data augmentation significantly improved generalisation

---

## 🛠️ Tech Stack

* Python
* PyTorch
* NumPy, Pandas
* Matplotlib

---

## 📂 Project Structure

```bash
02-skin-cancer-classification/
├── notebooks/
├── models/
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

* Increase dataset size for better generalisation
* Hyperparameter tuning
* Deploy model as a web-based diagnostic tool

