# Artificial Neural Networks (Lab) - Task # 02
### Shaheed Zulfikar Ali Bhutto Institute of Science & Technology (SZABIST), Islamabad
**Department of Robotics & Artificial Intelligence**

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-orange.svg)](https://scikit-learn.org/)
[![Flask](https://img.shields.io/badge/Flask-2.3%2B-lightgrey.svg)](https://flask.palletsprojects.com/)
[![Gradio](https://img.shields.io/badge/Gradio-3.5%2B-coral.svg)](https://www.gradio.app/)
[![License](https://img.shields.io/badge/License-Academic%20Use-green.svg)]()

---

## 📌 Student & Submission Details

| Field | Information |
| :--- | :--- |
| **Student Name** | **Taimur Sultan** |
| **Registration Number** | **24108145** |
| **Degree Program / Class** | BS (AI) - 5A |
| **Department** | Department of Robotics & Artificial Intelligence |
| **Institution** | SZABIST Islamabad Campus |
| **Course** | Artificial Neural Networks (Lab) |
| **Task Number** | Lab Task # 02 |
| **Instructor** | Mr. Hassaan Mujtaba |
| **Submission Report** | [`ANN_Lab_Task_2_24108145.pdf`](ANN_Lab_Task_2_24108145.pdf) / [` lab task 2 ann.pdf`](%20lab%20task%202%20ann.pdf) |
| **GitHub Repository** | [https://github.com/taimurkhaneth/LAB2-](https://github.com/taimurkhaneth/LAB2-) |

---

## 🚀 Project Overview

This repository provides the complete implementation, mathematical formulation, empirical evaluation, and production-grade interactive deployments for **Lab Task 2** of the Artificial Neural Networks course.

The primary objective of this lab is to build, validate, analyze, and deploy **Logistic Regression** models across two diverse classification paradigms:

1. **Task 1: Career Aptitude Prediction (Multi-Class Classification)**
   - Predicts an individual's recommended career path based on five core psychometric aptitude scores:
     - Numerical Aptitude
     - Spatial Aptitude
     - Perceptual Aptitude
     - Abstract Reasoning
     - Verbal Reasoning
   - Demonstrates high-cardinality multi-class classification and illustrates how extreme class sparsity affects linear decision boundaries.

2. **Task 2: Clinical Diabetes Risk Prediction (Binary Classification)**
   - Predicts the onset or diagnosis of diabetes ($0 = \text{No Diabetes}, 1 = \text{Diabetes}$) using patient biometrics and glycemic biomarkers:
     - Age
     - Body Mass Index (BMI)
     - Hemoglobin A1c Level (HbA1c)
     - Blood Glucose Level
   - Demonstrates high-accuracy binary classification on a massive clinical population ($N = 100,000$) with probabilistic risk calibration.

---

## 📐 Theoretical Framework: Logistic Regression

Logistic Regression is a foundational generalized linear model (GLM) for classification tasks. Unlike linear regression which outputs unbounded continuous numbers, logistic regression maps real-valued linear combinations to the interval $[0, 1]$.

### 1. The Standard Logistic (Sigmoid) Function
For binary classification ($y \in \{0, 1\}$), the hypothesis function predicts the posterior probability:
$$P(Y = 1 \mid X) = \sigma(z) = \frac{1}{1 + e^{-z}}$$
where $z = w^T X + b = w_0 + \sum_{i=1}^{m} w_i x_i$.

The decision rule with a standard threshold $\tau = 0.5$ is:
$$\hat{y} = \begin{cases} 1 & \text{if } P(Y=1 \mid X) \ge 0.5 \\ 0 & \text{if } P(Y=1 \mid X) < 0.5 \end{cases}$$

### 2. Binary Cross-Entropy Loss (Log-Loss)
Parameters $w$ and bias $b$ are estimated by minimizing the Negative Log-Likelihood (NLL):
$$\mathcal{L}_{BCE}(w, b) = -\frac{1}{N} \sum_{i=1}^{N} \left[ y^{(i)} \log \hat{y}^{(i)} + (1 - y^{(i)}) \log(1 - \hat{y}^{(i)}) \right]$$

### 3. Multinomial Logistic Regression (Softmax Function)
For $K$ mutually exclusive categories ($K > 2$), the model generalizes via the Softmax activation:
$$P(Y = k \mid X) = \frac{e^{w_k^T X + b_k}}{\sum_{j=1}^{K} e^{w_j^T X + b_j}}$$

Optimization is performed using L-BFGS or SAG/SAGA quasi-Newton solvers with Cross-Entropy Loss:
$$\mathcal{L}_{CCE} = -\frac{1}{N} \sum_{i=1}^{N} \sum_{k=1}^{K} y_{k}^{(i)} \log \hat{y}_{k}^{(i)}$$

---

## 📊 Summary of Model Evaluation & Results

| Metric | Task 1: Career Prediction | Task 2: Diabetes Prediction |
| :--- | :---: | :---: |
| **Dataset File** | `Data_final.csv` | `diabetes_prediction_dataset.csv` |
| **Classification Type** | Multi-Class Classification | Binary Classification |
| **Total Samples ($N$)** | 105 rows | 100,000 rows |
| **Unique Target Classes ($K$)** | 104 distinct career paths | 2 classes (0: Negative, 1: Positive) |
| **Feature Dimension ($m$)** | 5 numerical aptitude test scores | 4 continuous biometric indicators |
| **Train / Test Split** | 80% Train (84) / 20% Test (21) | 80% Train (80,000) / 20% Test (20,000) |
| **Solver / Max Iterations** | L-BFGS (`max_iter=1000`) | L-BFGS (`max_iter=1000`) |
| **Accuracy Score** | **0.00%** | **95.86%** |
| **Macro Average F1-Score** | 0.00 | 0.84 |
| **Weighted Average F1-Score** | 0.00 | 0.96 |
| **Primary Insight** | High class sparsity ($K \approx N$) | Exceptional discriminatory power & calibration |

### Detailed Analysis of Findings

1. **Task 1 Insights (Career Prediction)**:
   - With 105 total observations and 104 unique career labels, nearly every sample belongs to its own singleton class.
   - In a 20% test split ($N_{test} = 21$), each test record presents a unique category that had at most one instance in training.
   - The Scikit-Learn engine warns: `UserWarning: The number of unique classes is greater than 50% of the number of samples`.
   - **Academic Takeaway:** Illustrates the fundamental prerequisite of sample depth per class for parametric linear classifiers.

2. **Task 2 Insights (Diabetes Prediction)**:
   - The model learns from 80,000 training examples and achieves an outstanding **95.86% overall accuracy** on the unseen test set of 20,000 patients.
   - Precision for Class 0 (Non-Diabetic) reaches **96%** with **99% recall**.
   - Precision for Class 1 (Diabetic) reaches **88%** with a balanced F1-score of **0.71**, successfully isolating patients at clinical risk.
   - Probabilistic output allows fine-grained risk assessment (e.g., patient with elevated glucose and HbA1c yields a calibrated 65.39% probability of diabetes).

---

## 📁 Repository Structure

```text
TIM LAB TASK 2/
├── ANN_Lab_Task_2_24108145.ipynb   # Executed Jupyter Notebook with outputs & plots
├── task1_career.py                 # Standalone Python script for Task 1 (Career)
├── task2_diabetes.py               # Standalone Python script for Task 2 (Diabetes)
├── flask_career_app.py             # Dedicated Flask app for Task 1 (Port 5000)
├── flask_diabetes_app.py           # Dedicated Flask app for Task 2 (Port 5001)
├── app.py                          # Unified Dual-Model Dashboard (Port 5000 / 5050)
├── Data_final.csv                  # Dataset for Task 1 (Aptitude scores & careers)
├── diabetes_prediction_dataset.csv # Dataset for Task 2 (Clinical diabetes records)
├── ANN_Lab_Task_2_24108145.pdf     # Official submitted lab report PDF
├──  lab task 2 ann.pdf             # Original submitted lab report PDF
├── requirements.txt                # Python package dependencies
├── .gitignore                      # Git configuration & cache filters
└── README.md                       # Comprehensive documentation
```

---

## 🛠️ Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/taimurkhaneth/LAB2-.git
cd LAB2-
```

### 2. Set Up Virtual Environment (Recommended)
```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On macOS / Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🖥️ How to Run the Applications

### Option A: Standalone Console Execution
Run the standalone scripts to view model training logs, evaluation metrics, and sample inferences in your terminal:
```bash
# Run Task 1: Career Prediction
python3 task1_career.py

# Run Task 2: Diabetes Prediction
python3 task2_diabetes.py
```

### Option B: Dedicated Flask Web Applications
Run each task as an independent standalone Flask server:
```bash
# Launch Task 1 Career Predictor on http://127.0.0.1:5000
python3 flask_career_app.py

# Launch Task 2 Diabetes Predictor on http://127.0.0.1:5001
python3 flask_diabetes_app.py
```

### Option C: Unified Master Dashboard (`app.py`)
Launch the unified web platform hosting both classification models simultaneously:
```bash
python3 app.py
```
Open your browser and navigate to:  
👉 **[http://127.0.0.1:5000](http://127.0.0.1:5000)**

### Option D: Jupyter Notebook
To inspect or re-execute the interactive Jupyter Notebook:
```bash
jupyter notebook ANN_Lab_Task_2_24108145.ipynb
```

---

## 🌐 REST API Endpoints

The unified dashboard (`app.py`) includes JSON REST APIs for integration with external systems or mobile apps:

### 1. Career Prediction API
- **Endpoint:** `POST /api/predict/career`
- **Request Body:**
```json
{
  "numerical": 20,
  "spatial": 16,
  "perceptual": 6,
  "abstract": 26,
  "verbal": 7
}
```
- **Response:**
```json
{
  "predicted_career": "Research Scientist"
}
```

### 2. Diabetes Risk Prediction API
- **Endpoint:** `POST /api/predict/diabetes`
- **Request Body:**
```json
{
  "age": 15,
  "bmi": 14,
  "hba1c": 11,
  "glucose": 12
}
```
- **Response:**
```json
{
  "prediction": "Diabetic",
  "probability_percent": 65.39
}
```

---

## 👨‍💻 Author & Acknowledgements

- **Student:** Taimur Sultan (Registration: `24108145`)
- **Institution:** Shaheed Zulfikar Ali Bhutto Institute of Science & Technology (SZABIST), Islamabad Campus
- **Department:** Department of Robotics & Artificial Intelligence
- **Lab Instructor:** Mr. Hassaan Mujtaba
- **Repository:** [https://github.com/taimurkhaneth/LAB2-](https://github.com/taimurkhaneth/LAB2-)
