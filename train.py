import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import joblib
import os

def train_cart_model(csv_path):
    df = pd.read_csv(csv_path)
    df = df.dropna()
    X = df.drop(columns=["filename", "target", "error"], errors="ignore")
    y = df["target"].map({"abnormal": 1, "normal": 0})

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = DecisionTreeClassifier(criterion="gini", max_depth=5, random_state=42)
    model.fit(X_train, y_train)

    # Save the model and features
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/model.pkl")
    joblib.dump(X.columns.tolist(), "models/features.pkl")

    print("✅ Model saved to:", os.path.abspath("models/model.pkl"))
    print("✅ Features saved to:", os.path.abspath("models/features.pkl"))

if __name__ == "__main__":
    train_cart_model("ecg_combined_dataset.csv")  # Make sure this CSV exists
