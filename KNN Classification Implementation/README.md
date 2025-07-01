# 🌸 K-Nearest Neighbors (KNN) Classification – Iris Dataset

This project demonstrates how to use the **K-Nearest Neighbors (KNN)** algorithm for classifying flower species using the famous **Iris dataset**. It includes data normalization, model training, evaluation with accuracy and confusion matrix, and a decision-making guide.

---

## 📁 Project Structure

```
KNN_Classification_Iris/
├── data/
│   └── Iris.csv                  # Dataset
├── scripts/
│   └── knn_classifier.py         # Main KNN script
├── notebooks/
│   └── KNN_Exploration.ipynb     # Interactive Jupyter notebook
├── outputs/
│   └── confusion_matrix.png      # Confusion matrix plot (generated after running script)
├── requirements.txt              # Dependencies
└── README.md                     # Project documentation
```

---

## 🚀 How to Run

### 1. ✅ Install Python 3.7+

Ensure Python is installed:
```bash
python --version
```

---

### 2. 🛠 Install Dependencies

Create a virtual environment (optional but recommended):
```bash
python -m venv venv
# Activate (Windows)
venv\Scripts\activate
# Activate (Mac/Linux)
source venv/bin/activate
```

Then install dependencies:
```bash
pip install -r requirements.txt
```

---

### 3. ▶️ Run the Classifier Script

```bash
cd scripts
python knn_classifier.py
```

- Displays accuracy for K = 1 to 10
- Saves a confusion matrix plot to `outputs/confusion_matrix.png`

---

### 4. 💻 Explore via Jupyter Notebook

For step-by-step interactive execution:

```bash
cd notebooks
jupyter notebook
```

Open `knn_exploration.ipynb` and run all cells.

---

## 📊 What You'll Learn

- How KNN works
- How to normalize data
- Model accuracy evaluation for different K values
- Confusion matrix visualization

---

## 📚 Dataset

- **Iris Dataset**: Contains measurements (sepal/petal length & width) for 3 flower species.

Source: [UCI Machine Learning Repository – Iris](https://www.kaggle.com/datasets/uciml/iris)

---

## 🛡 Requirements

See `requirements.txt`:

```text
pandas
numpy
matplotlib
scikit-learn
```

---