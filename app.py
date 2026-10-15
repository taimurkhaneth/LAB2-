"""
Artificial Neural Networks (ANN) - Lab Task 02
Unified Web Portal (Port 5000)

Student Name: Taimur Sultan
Registration Number: 24108145
Class: BS (AI) - 5A
Instructor: Mr. Hassaan Mujtaba
Institution: SZABIST Islamabad
"""

import os
import pandas as pd
from flask import Flask, request, render_template_string, jsonify
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

app = Flask(__name__)

# ==============================================================================
# MODEL 1: Career Prediction (Multi-Class Logistic Regression)
# ==============================================================================
print("Loading Model 1 (Career Prediction)...")
df1 = pd.read_csv("Data_final.csv")
feature_cols_1 = [
    "Numerical Aptitude",
    "Spatial Aptitude",
    "Perceptual Aptitude",
    "Abstract Reasoning",
    "Verbal Reasoning",
]
X1 = df1[feature_cols_1]
y1 = df1["Career"]

X_train1, X_test1, y_train1, y_test1 = train_test_split(
    X1, y1, test_size=0.2, random_state=42
)
model_career = LogisticRegression(max_iter=1000)
model_career.fit(X_train1, y_train1)
acc_career = accuracy_score(y_test1, model_career.predict(X_test1))


def predict_career(numerical, spatial, perceptual, abstract, verbal):
    data = pd.DataFrame(
        [[numerical, spatial, perceptual, abstract, verbal]],
        columns=X1.columns,
    )
    return model_career.predict(data)[0]


# ==============================================================================
# MODEL 2: Diabetes Prediction (Binary Logistic Regression)
# ==============================================================================
print("Loading Model 2 (Diabetes Prediction)...")
df2 = pd.read_csv("diabetes_prediction_dataset.csv")
feature_cols_2 = ["age", "bmi", "HbA1c_level", "blood_glucose_level"]
X2 = df2[feature_cols_2]
y2 = df2["diabetes"]

X_train2, X_test2, y_train2, y_test2 = train_test_split(
    X2, y2, test_size=0.2, random_state=42
)
model_diabetes = LogisticRegression(max_iter=1000)
model_diabetes.fit(X_train2, y_train2)
acc_diabetes = accuracy_score(y_test2, model_diabetes.predict(X_test2))


def predict_diabetes(age, bmi, hba1c, glucose):
    data = pd.DataFrame([[age, bmi, hba1c, glucose]], columns=X2.columns)
    pred = model_diabetes.predict(data)[0]
    prob = model_diabetes.predict_proba(data)[0][1]
    label = "Diabetic" if pred == 1 else "Not Diabetic"
    return label, round(prob * 100, 2)


# ==============================================================================
# UNIFIED HTML TEMPLATE
# ==============================================================================
UNIFIED_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ANN Lab Task 02 - Unified Portal | Taimur Sultan (24108145)</title>
    <style>
        :root {
            --bg: #090d16;
            --card-bg: rgba(18, 24, 38, 0.75);
            --card-border: rgba(255, 255, 255, 0.08);
            --text-primary: #f8fafc;
            --text-secondary: #94a3b8;
            --accent-orange: #f97316;
            --accent-cyan: #06b6d4;
            --accent-green: #10b981;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
        body {
            background: radial-gradient(circle at 20% 20%, rgba(249, 115, 22, 0.08) 0%, transparent 40%),
                        radial-gradient(circle at 80% 80%, rgba(6, 182, 212, 0.08) 0%, transparent 40%),
                        #090d16;
            color: var(--text-primary);
            min-height: 100vh;
            padding: 30px 16px;
        }
        .wrapper {
            max-width: 1080px;
            margin: 0 auto;
        }
        header {
            text-align: center;
            margin-bottom: 36px;
        }
        .institution-badge {
            display: inline-block;
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid var(--card-border);
            padding: 6px 16px;
            border-radius: 9999px;
            font-size: 13px;
            color: #cbd5e1;
            margin-bottom: 14px;
            letter-spacing: 0.04em;
        }
        h1 {
            font-size: 32px;
            font-weight: 800;
            background: linear-gradient(135deg, #ffffff 30%, #94a3b8 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 8px;
        }
        .header-sub {
            color: var(--text-secondary);
            font-size: 15px;
        }
        .student-bar {
            display: flex;
            justify-content: center;
            gap: 20px;
            flex-wrap: wrap;
            margin-top: 16px;
            font-size: 13px;
        }
        .student-bar span {
            background: rgba(30, 41, 59, 0.6);
            padding: 6px 14px;
            border-radius: 8px;
            border: 1px solid var(--card-border);
        }
        .student-bar b { color: #f8fafc; }
        .tabs {
            display: flex;
            gap: 12px;
            justify-content: center;
            margin-bottom: 28px;
        }
        .tab-btn {
            background: rgba(30, 41, 59, 0.6);
            border: 1px solid var(--card-border);
            color: var(--text-secondary);
            padding: 12px 24px;
            border-radius: 12px;
            font-size: 15px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.25s ease;
        }
        .tab-btn:hover {
            color: var(--text-primary);
            border-color: rgba(255, 255, 255, 0.2);
        }
        .tab-btn.active.career {
            background: rgba(249, 115, 22, 0.15);
            border-color: var(--accent-orange);
            color: #fb923c;
        }
        .tab-btn.active.diabetes {
            background: rgba(6, 182, 212, 0.15);
            border-color: var(--accent-cyan);
            color: #22d3ee;
        }
        .tab-content { display: none; }
        .tab-content.active { display: block; }

        .card {
            background: var(--card-bg);
            border: 1px solid var(--card-border);
            border-radius: 20px;
            backdrop-filter: blur(20px);
            padding: 32px;
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.4);
        }
        .card-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 24px;
            flex-wrap: wrap;
            gap: 12px;
        }
        .card-title { font-size: 22px; font-weight: 700; }
        .stat-pill {
            padding: 6px 14px;
            border-radius: 9999px;
            font-size: 13px;
            font-weight: 600;
        }
        .pill-orange { background: rgba(249, 115, 22, 0.15); color: #fb923c; border: 1px solid rgba(249, 115, 22, 0.3); }
        .pill-cyan { background: rgba(6, 182, 212, 0.15); color: #22d3ee; border: 1px solid rgba(6, 182, 212, 0.3); }

        .form-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 18px;
            margin-bottom: 24px;
        }
        .form-group {
            display: flex;
            flex-direction: column;
            gap: 6px;
        }
        label {
            font-size: 13px;
            font-weight: 500;
            color: #cbd5e1;
        }
        input {
            background: rgba(15, 23, 42, 0.6);
            border: 1px solid var(--card-border);
            border-radius: 10px;
            padding: 12px 14px;
            color: #fff;
            font-size: 14px;
            transition: all 0.2s;
        }
        input:focus {
            outline: none;
            border-color: #38bdf8;
            box-shadow: 0 0 0 3px rgba(56, 189, 248, 0.15);
        }
        .btn {
            width: 100%;
            padding: 14px;
            border-radius: 10px;
            border: none;
            font-size: 15px;
            font-weight: 700;
            cursor: pointer;
            transition: transform 0.2s, opacity 0.2s;
        }
        .btn:hover { transform: translateY(-2px); opacity: 0.95; }
        .btn-orange { background: var(--accent-orange); color: #fff; }
        .btn-cyan { background: var(--accent-cyan); color: #090d16; }

        .result-box {
            margin-top: 24px;
            padding: 20px;
            border-radius: 12px;
            text-align: center;
        }
        .result-career {
            background: rgba(249, 115, 22, 0.12);
            border: 1px solid rgba(249, 115, 22, 0.3);
            color: #fed7aa;
        }
        .result-diabetes {
            background: rgba(6, 182, 212, 0.12);
            border: 1px solid rgba(6, 182, 212, 0.3);
            color: #a5f3fc;
        }
        .result-box h3 { font-size: 20px; margin-bottom: 4px; }
        .result-box p { font-size: 14px; opacity: 0.85; }

        footer {
            margin-top: 40px;
            text-align: center;
            color: var(--text-secondary);
            font-size: 13px;
        }
    </style>
</head>
<body>
    <div class="wrapper">
        <header>
            <div class="institution-badge">SZABIST ISLAMABAD &bull; DEPARTMENT OF ROBOTICS & AI</div>
            <h1>Artificial Neural Networks - Lab Task # 02</h1>
            <p class="header-sub">Logistic Regression Classification Models &bull; Interactive Web Platform</p>
            <div class="student-bar">
                <span>Student: <b>Taimur Sultan</b></span>
                <span>Reg No: <b>24108145</b></span>
                <span>Class: <b>BS (AI) - 5A</b></span>
                <span>Instructor: <b>Mr. Hassaan Mujtaba</b></span>
            </div>
        </header>

        <div class="tabs">
            <button class="tab-btn career {% if active_tab == 'career' %}active{% endif %}" onclick="showTab('career')">
                Task 1: Career Prediction
            </button>
            <button class="tab-btn diabetes {% if active_tab == 'diabetes' %}active{% endif %}" onclick="showTab('diabetes')">
                Task 2: Diabetes Prediction
            </button>
        </div>

        <!-- TAB 1: CAREER PREDICTION -->
        <div id="tab-career" class="tab-content {% if active_tab == 'career' %}active{% endif %}">
            <div class="card">
                <div class="card-header">
                    <div>
                        <h2 class="card-title">Career Aptitude Predictor</h2>
                        <p style="color: var(--text-secondary); font-size: 13px; margin-top: 4px;">
                            Multi-Class Logistic Regression &bull; Dataset: <code>Data_final.csv</code>
                        </p>
                    </div>
                    <span class="stat-pill pill-orange">Max Iter: 1000 &bull; Random State: 42</span>
                </div>

                <form method="post" action="/predict/career">
                    <div class="form-grid">
                        <div class="form-group">
                            <label>Numerical Aptitude (e.g. 20):</label>
                            <input type="number" step="any" name="numerical" value="{{ career_vals.get('numerical', '20') }}" required>
                        </div>
                        <div class="form-group">
                            <label>Spatial Aptitude (e.g. 16):</label>
                            <input type="number" step="any" name="spatial" value="{{ career_vals.get('spatial', '16') }}" required>
                        </div>
                        <div class="form-group">
                            <label>Perceptual Aptitude (e.g. 6):</label>
                            <input type="number" step="any" name="perceptual" value="{{ career_vals.get('perceptual', '6') }}" required>
                        </div>
                        <div class="form-group">
                            <label>Abstract Reasoning (e.g. 26):</label>
                            <input type="number" step="any" name="abstract" value="{{ career_vals.get('abstract', '26') }}" required>
                        </div>
                        <div class="form-group">
                            <label>Verbal Reasoning (e.g. 7):</label>
                            <input type="number" step="any" name="verbal" value="{{ career_vals.get('verbal', '7') }}" required>
                        </div>
                    </div>
                    <button type="submit" class="btn btn-orange">Predict Career Path</button>
                </form>

                {% if career_result %}
                <div class="result-box result-career">
                    <h3>{{ career_result }}</h3>
                    <p>Predicted using Multinomial Logistic Regression model</p>
                </div>
                {% endif %}
            </div>
        </div>

        <!-- TAB 2: DIABETES PREDICTION -->
        <div id="tab-diabetes" class="tab-content {% if active_tab == 'diabetes' %}active{% endif %}">
            <div class="card">
                <div class="card-header">
                    <div>
                        <h2 class="card-title">Diabetes Risk Predictor</h2>
                        <p style="color: var(--text-secondary); font-size: 13px; margin-top: 4px;">
                            Binary Logistic Regression &bull; Dataset: <code>diabetes_prediction_dataset.csv</code>
                        </p>
                    </div>
                    <span class="stat-pill pill-cyan">Accuracy: ~95.86% &bull; Samples: 100,000</span>
                </div>

                <form method="post" action="/predict/diabetes">
                    <div class="form-grid">
                        <div class="form-group">
                            <label>Age (years, e.g. 15):</label>
                            <input type="number" step="any" name="age" value="{{ diabetes_vals.get('age', '15') }}" required>
                        </div>
                        <div class="form-group">
                            <label>BMI (kg/m², e.g. 14):</label>
                            <input type="number" step="any" name="bmi" value="{{ diabetes_vals.get('bmi', '14') }}" required>
                        </div>
                        <div class="form-group">
                            <label>HbA1c Level (e.g. 11):</label>
                            <input type="number" step="any" name="hba1c" value="{{ diabetes_vals.get('hba1c', '11') }}" required>
                        </div>
                        <div class="form-group">
                            <label>Blood Glucose Level (mg/dL, e.g. 12):</label>
                            <input type="number" step="any" name="glucose" value="{{ diabetes_vals.get('glucose', '12') }}" required>
                        </div>
                    </div>
                    <button type="submit" class="btn btn-cyan">Assess Diabetes Risk</button>
                </form>

                {% if diabetes_result %}
                <div class="result-box result-diabetes">
                    <h3>{{ diabetes_result }}</h3>
                    <p>Estimated by Binary Logistic Sigmoid Classifier with Probability Thresholding</p>
                </div>
                {% endif %}
            </div>
        </div>

        <footer>
            SZABIST Islamabad Campus &bull; Artificial Neural Networks Lab Task 02 &bull; Student: Taimur Sultan (24108145)
        </footer>
    </div>

    <script>
        function showTab(tabName) {
            document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
            document.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active'));
            document.getElementById('tab-' + tabName).classList.add('active');
            event.target.classList.add('active');
        }
    </script>
</body>
</html>
"""


@app.route("/", methods=["GET"])
def index():
    return render_template_string(
        UNIFIED_HTML,
        active_tab="career",
        career_vals={},
        diabetes_vals={},
        career_result=None,
        diabetes_result=None,
    )


@app.route("/predict/career", methods=["POST"])
def route_predict_career():
    try:
        numerical = float(request.form.get("numerical", 20))
        spatial = float(request.form.get("spatial", 16))
        perceptual = float(request.form.get("perceptual", 6))
        abstract = float(request.form.get("abstract", 26))
        verbal = float(request.form.get("verbal", 7))
        pred = predict_career(numerical, spatial, perceptual, abstract, verbal)
        res_text = f"Predicted Career: {pred}"
    except Exception as e:
        res_text = f"Error: {str(e)}"

    return render_template_string(
        UNIFIED_HTML,
        active_tab="career",
        career_vals=request.form,
        diabetes_vals={},
        career_result=res_text,
        diabetes_result=None,
    )


@app.route("/predict/diabetes", methods=["POST"])
def route_predict_diabetes():
    try:
        age = float(request.form.get("age", 15))
        bmi = float(request.form.get("bmi", 14))
        hba1c = float(request.form.get("hba1c", 11))
        glucose = float(request.form.get("glucose", 12))
        label, prob = predict_diabetes(age, bmi, hba1c, glucose)
        res_text = f"Prediction: {label} (probability of diabetes: {prob}%)"
    except Exception as e:
        res_text = f"Error: {str(e)}"

    return render_template_string(
        UNIFIED_HTML,
        active_tab="diabetes",
        career_vals={},
        diabetes_vals=request.form,
        career_result=None,
        diabetes_result=res_text,
    )


@app.route("/api/predict/career", methods=["POST"])
def api_career():
    data = request.get_json(force=True)
    pred = predict_career(
        float(data.get("numerical", 20)),
        float(data.get("spatial", 16)),
        float(data.get("perceptual", 6)),
        float(data.get("abstract", 26)),
        float(data.get("verbal", 7)),
    )
    return jsonify({"predicted_career": pred})


@app.route("/api/predict/diabetes", methods=["POST"])
def api_diabetes():
    data = request.get_json(force=True)
    label, prob = predict_diabetes(
        float(data.get("age", 15)),
        float(data.get("bmi", 14)),
        float(data.get("hba1c", 11)),
        float(data.get("glucose", 12)),
    )
    return jsonify({"prediction": label, "probability_percent": prob})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"Starting Unified ANN Lab Task 2 Portal on http://127.0.0.1:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
