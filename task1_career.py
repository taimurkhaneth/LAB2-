"""
Artificial Neural Networks (ANN) - Lab Task 02
Task 1: Career Prediction using Logistic Regression

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
    print("TASK 1: Career Prediction using Logistic Regression")
    print("=" * 70)

    # 1. Load the data
    df = pd.read_csv("Data_final.csv")
    print("\nDataset Info:")
    print(f"Total Samples: {df.shape[0]}, Total Features: {df.shape[1]}")
    print("\nFirst 5 rows:")
    print(df.head())

    # 2. Choose input columns (X) and target column (y)
    feature_cols = [
        "Numerical Aptitude",
        "Spatial Aptitude",
        "Perceptual Aptitude",
        "Abstract Reasoning",
        "Verbal Reasoning",
    ]
    X = df[feature_cols]
    y = df["Career"]  # categorical target -> multi-class classification

    print("\nTarget Column: Career")
    print(f"Number of unique career categories: {y.nunique()}")

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
    print(f"\nAccuracy: {accuracy}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    # 6. Prediction function (used by both Gradio and Flask)
    def predict(numerical, spatial, perceptual, abstract, verbal):
        data = pd.DataFrame(
            [[numerical, spatial, perceptual, abstract, verbal]],
            columns=X.columns,
        )
        return model.predict(data)[0]

    # Test Sample from Lab Report: (20, 16, 6, 26, 7)
    sample_inputs = (20.0, 16.0, 6.0, 26.0, 7.0)
    sample_pred = predict(*sample_inputs)
    print("=" * 70)
    print("Sample Inference (from Lab Report):")
    print(f"Inputs: Numerical={sample_inputs[0]}, Spatial={sample_inputs[1]}, "
          f"Perceptual={sample_inputs[2]}, Abstract={sample_inputs[3]}, Verbal={sample_inputs[4]}")
    print(f"Predicted Career: {sample_pred}")
    print("=" * 70)

    return model, predict


if __name__ == "__main__":
    main()
