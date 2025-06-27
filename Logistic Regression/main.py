import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_auc_score, roc_curve, accuracy_score
)

# Create folder for plots
os.makedirs("plots", exist_ok=True)

# Load CSV
data = pd.read_csv("data.csv")

# Drop unwanted columns
data = data.drop(columns=["id", "Unnamed: 32"], errors='ignore')

# Convert 'diagnosis' to binary: M -> 1, B -> 0
data["target"] = data["diagnosis"].map({"M": 1, "B": 0})
data = data.drop(columns=["diagnosis"])

# Confirm data shape
print("Dataset shape:", data.shape)
print("Columns:", list(data.columns))
print("\nPreview of data:")
print(data.head())

# Split features and target
X = data.drop(columns=["target"])
y = data["target"]

# Train-test split and scaling
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train logistic regression model
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_scaled, y_train)

# Predict and Evaluate
y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]

print("\n--- Evaluation Metrics ---")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("ROC AUC Score:", roc_auc_score(y_test, y_prob))
print("Accuracy Score:", accuracy_score(y_test, y_pred))

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc_score(y_test, y_prob):.2f})")
plt.plot([0, 1], [0, 1], 'k--', label="Random")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.grid(True)
plt.savefig("plots/roc_curve.png")
plt.close()

# Threshold Tuning
threshold = 0.6
y_pred_custom = (y_prob >= threshold).astype(int)
print(f"\n--- Threshold Tuning (threshold = {threshold}) ---")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_custom))
print("Classification Report:\n", classification_report(y_test, y_pred_custom))

# Sigmoid Function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

z = np.linspace(-10, 10, 100)
sig = sigmoid(z)

plt.figure(figsize=(8, 5))
plt.plot(z, sig, label="Sigmoid Function")
plt.xlabel("z")
plt.ylabel("sigmoid(z)")
plt.title("Sigmoid Activation Function")
plt.grid(True)
plt.legend()
plt.savefig("plots/sigmoid.png")
plt.close()

print("\n✅ Done! All plots saved in the 'plots/' folder.")
