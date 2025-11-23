# model_training.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
import joblib

def train_and_save_model(csv_path="gdl_synthetic_data.csv"):
    df = pd.read_csv(csv_path)

    # Convert continuous target into categories
    bins = [0, 40, 70, 100]
    labels = ['Low', 'Medium', 'High']
    df['Performance_Category'] = pd.cut(df['Performance_Score'], bins=bins, labels=labels)

    # Features & target
    X = df[['Porosity', 'Pore_Size_um', 'Fiber_Arrangement_Angle', 'Wettability_Contact_Angle']]
    y = df['Performance_Category']

    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    # Naive Bayes model
    model = GaussianNB()
    model.fit(X_train, y_train)

    # Save scaler & model
    joblib.dump(scaler, "scaler.pkl")
    joblib.dump(model, "model.pkl")

    print("Model and scaler saved successfully.")

    return model

if __name__ == "__main__":
    train_and_save_model()
