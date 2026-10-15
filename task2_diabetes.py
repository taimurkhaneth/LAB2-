"""
Artificial Neural Networks (ANN) - Lab Task 02
Task 2: Diabetes Prediction using Logistic Regression

Student Name: Taimur Sultan
Registration Number: 24108145
Class: BS (AI) - 5A
Instructor: Mr. Hassaan Mujtaba
Institution: SZABIST Islamabad
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


def main():
    print("=" * 70)
    print("TASK 2: Diabetes Prediction using Logistic Regression")
    print("=" * 70)

    # 1. Load the data
    df = pd.read_csv("diabetes_prediction_dataset.csv")
    print("\nDataset Info:")
    print(f"Total Samples: {df.shape[0]}, Total Features: {df.shape[1]}")
    print("\nFirst 5 rows:")
    print(df.head())

    # 2. Choose input columns (X) and target column (y)
    feature_cols = ["age", "bmi", "HbA1c_level", "blood_glucose_level"]
    X = df[feature_cols]
    y = df["diabetes"]  # 0 = No Diabetes, 1 = Diabetes

    print("\nTarget Class Distribution:")
    print(y.value_counts())

    # 3. Split into training and testing data (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"\nTraining set size: {X_train.shape[0]}")
    print(f"Testing set size: {X_test.shape[0]}")

    # 4. Train the Logistic Regression model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # 5. Check how good the model is
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # 6. Prediction function (used by both Gradio and Flask)
    def predict(age, bmi, hba1c, glucose):
        data = pd.DataFrame([[age, bmi, hba1c, glucose]], columns=X.columns)
        result = model.predict(data)[0]
        prob = model.predict_proba(data)[0][1]
        label = "Diabetic" if result == 1 else "Not Diabetic"
        return f"{label} (probability of diabetes: {round(prob * 100, 2)}%)"

    # Test Sample from Lab Report: (age=15, bmi=14, hba1c=11, glucose=12)
    sample_inputs = (15.0, 14.0, 11.0, 12.0)
    sample_res = predict(*sample_inputs)
    print("=" * 70)
    print("Sample Inference (from Lab Report):")
    print(f"Inputs: Age={sample_inputs[0]}, BMI={sample_inputs[1]}, "
          f"HbA1c={sample_inputs[2]}, Glucose={sample_inputs[3]}")
    print(f"Prediction: {sample_res}")
    print("=" * 70)

    return model, predict


if __name__ == "__main__":
    main()
