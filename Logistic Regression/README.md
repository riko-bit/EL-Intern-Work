# Logistic Regression Binary Classifier

This project implements a binary classification model using **Logistic Regression** on the **Breast Cancer Wisconsin Dataset (CSV format)**.

## 🔍 Objective
Build a classifier to predict whether a tumor is **malignant (M)** or **benign (B)** based on various cell nucleus features.

---

## 📁 Project Structure
```
.
├── data.csv                  # Input dataset (you provide this)
├── main.py                  # Main Python script
├── requirements.txt         # Python dependencies
├── README.md                # Project overview
└── plots/
    ├── roc_curve.png        # ROC curve plot
    └── sigmoid.png          # Sigmoid activation function plot
```

---

## 🚀 How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the classifier
```bash
python main.py
```

Make sure `data.csv` is in the **same directory** as `main.py`.

---

## 🧠 Features of the Script
- Preprocesses your dataset:
  - Drops `id` and `Unnamed: 32` columns
  - Converts `diagnosis` column into binary labels (`M=1`, `B=0`)
- Splits data into training/testing sets
- Standardizes features with `StandardScaler`
- Trains a `LogisticRegression` model
- Evaluates with:
  - Confusion matrix
  - Classification report
  - ROC-AUC score
- Plots:
  - ROC curve
  - Sigmoid activation function
- Performs threshold tuning at 0.6 and re-evaluates model

---

## 📊 Example Output
Evaluation metrics and plots (like ROC Curve and Sigmoid Function) will be saved under the `plots/` folder after successful execution.

---

## 📌 Notes
- Dataset: Breast Cancer Diagnostic data (from `data.csv`)
- Target column: `diagnosis` (`M` or `B`)
- Converted column: `target` (1 or 0)

---

## 🧪 Requirements
See [`requirements.txt`](requirements.txt)
