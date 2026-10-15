"""
Artificial Neural Networks (ANN) - Lab Task 02
Standalone Flask Application - Task 2: Diabetes Prediction (Port 5001)

Student Name: Taimur Sultan
Registration Number: 24108145
Class: BS (AI) - 5A
SZABIST Islamabad
"""

import os
import pandas as pd
from flask import Flask, request, render_template_string
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)

# Load dataset and train model
df = pd.read_csv("diabetes_prediction_dataset.csv")
X = df[["age", "bmi", "HbA1c_level", "blood_glucose_level"]]
y = df["diabetes"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)


def predict(age, bmi, hba1c, glucose):
    data = pd.DataFrame([[age, bmi, hba1c, glucose]], columns=X.columns)
    result = model.predict(data)[0]
    prob = model.predict_proba(data)[0][1]
    label = "Diabetic" if result == 1 else "Not Diabetic"
    return f"{label} (probability of diabetes: {round(prob * 100, 2)}%)"


HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Diabetes Predictor - SZABIST ANN Lab 2</title>
    <style>
        :root {
            --bg: #0f172a;
            --card-bg: rgba(30, 41, 59, 0.7);
            --border: rgba(255, 255, 255, 0.1);
            --primary: #06b6d4;
            --primary-hover: #0891b2;
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
        body {
            background: linear-gradient(135deg, #090d16 0%, #0f172a 50%, #0e3b43 100%);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 24px;
            color: var(--text-main);
        }
        .container {
            width: 100%;
            max-width: 580px;
            background: var(--card-bg);
            backdrop-filter: blur(16px);
            border: 1px solid var(--border);
            border-radius: 20px;
            padding: 36px;
            box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
        }
        .badge {
            display: inline-block;
            background: rgba(6, 182, 212, 0.15);
            color: #22d3ee;
            border: 1px solid rgba(6, 182, 212, 0.3);
            font-size: 12px;
            font-weight: 600;
            padding: 4px 12px;
            border-radius: 9999px;
            margin-bottom: 12px;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        h2 { font-size: 24px; font-weight: 700; margin-bottom: 6px; }
        .subtitle { color: var(--text-muted); font-size: 14px; margin-bottom: 24px; }
        .form-group { margin-bottom: 16px; }
        label { display: block; font-size: 13px; font-weight: 500; margin-bottom: 6px; color: #cbd5e1; }
        input {
            width: 100%;
            padding: 12px 14px;
            background: rgba(15, 23, 42, 0.6);
            border: 1px solid var(--border);
            border-radius: 10px;
            color: #fff;
            font-size: 14px;
            transition: border-color 0.2s;
        }
        input:focus { outline: none; border-color: var(--primary); }
        .btn-submit {
            width: 100%;
            background: var(--primary);
            color: #0f172a;
            font-size: 15px;
            font-weight: 700;
            padding: 13px;
            border: none;
            border-radius: 10px;
            cursor: pointer;
            transition: all 0.2s;
            margin-top: 10px;
        }
        .btn-submit:hover { background: var(--primary-hover); color: #fff; transform: translateY(-1px); }
        .result-box {
            margin-top: 24px;
            padding: 18px;
            background: rgba(6, 182, 212, 0.12);
            border: 1px solid rgba(6, 182, 212, 0.3);
            border-radius: 12px;
            text-align: center;
        }
        .result-box h3 { font-size: 18px; color: #a5f3fc; }
        .footer { margin-top: 24px; text-align: center; font-size: 12px; color: var(--text-muted); }
    </style>
</head>
<body>
    <div class="container">
        <span class="badge">SZABIST ANN LAB 2 &bull; TASK 2</span>
        <h2>Diabetes Predictor (Logistic Regression)</h2>
        <p class="subtitle">Binary Logistic Regression &bull; Student: Taimur Sultan (24108145)</p>

        <form method="post">
            <div class="form-group">
                <label>Age:</label>
                <input type="number" step="any" name="age" value="{{ request.form.get('age', '15') }}" required>
            </div>
            <div class="form-group">
                <label>BMI:</label>
                <input type="number" step="any" name="bmi" value="{{ request.form.get('bmi', '14') }}" required>
            </div>
            <div class="form-group">
                <label>HbA1c Level:</label>
                <input type="number" step="any" name="hba1c" value="{{ request.form.get('hba1c', '11') }}" required>
            </div>
            <div class="form-group">
                <label>Blood Glucose Level:</label>
                <input type="number" step="any" name="glucose" value="{{ request.form.get('glucose', '12') }}" required>
            </div>
            <button type="submit" class="btn-submit">Predict Diabetes</button>
        </form>

        {% if result %}
        <div class="result-box">
            <h3>{{ result }}</h3>
        </div>
        {% endif %}

        <div class="footer">
            Accuracy: ~95.86% &bull; SZABIST Islamabad
        </div>
    </div>
</body>
</html>
"""


@app.route("/", methods=["GET", "POST"])
def home():
    result = ""
    if request.method == "POST":
        try:
            age = float(request.form["age"])
            bmi = float(request.form["bmi"])
            hba1c = float(request.form["hba1c"])
            glucose = float(request.form["glucose"])
            predicted = predict(age, bmi, hba1c, glucose)
            result = f"Prediction: {predicted}"
        except Exception as e:
            result = f"Error: {str(e)}"
    return render_template_string(HTML_TEMPLATE, result=result)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    print(f"Starting Task 2 Diabetes Flask App on http://127.0.0.1:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
