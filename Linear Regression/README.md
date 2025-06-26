# Linear Regression Project - House Price Prediction 🏠📈

## ✅ Objective
To implement and understand both **Simple** and **Multiple Linear Regression** using Scikit-learn on a real-world housing dataset.

---

## 📂 Folder Structure
linear_regression_project/
├── data/
│ └── Housing.csv # Dataset
├── linear_regression.py # Final Python script
├── Linear_Regression_Housing.ipynb # Jupyter notebook version
├── README.md # This file
├── requirements.txt # Required packages


---

## 📊 Dataset
- File: `Housing.csv`
- Columns used:
  - `area` (sq ft)
  - `bedrooms`
  - `bathrooms`
  - `stories`
  - `price` (target variable)

---

## 🛠️ Tools Used
- Python
- Pandas
- Scikit-learn
- Matplotlib
- Jupyter Notebook

---

## 🚀 Steps Performed

1. **Import and preprocess the dataset**
   - Loaded using Pandas and cleaned missing data.
2. **Split data into train-test sets**
   - 80% training and 20% testing using `train_test_split`.
3. **Fit Linear Regression model**
   - Used `LinearRegression()` from Scikit-learn.
4. **Evaluate the model**
   - MAE, MSE, and R² Score calculated.
5. **Plot the regression line**
   - Visualized actual vs predicted values.
6. **Multiple Linear Regression**
   - Added more predictors to improve model performance.

---

## 📈 Evaluation Metrics

- **MAE**: Mean Absolute Error
- **MSE**: Mean Squared Error
- **R² Score**: Coefficient of determination

---

## 🧪 How to Run

1. Install dependencies:
pip install -r requirements.txt

2. Run Python script:
python linear_regression.py

3. Or open the Jupyter notebook:
jupyter notebook Linear_Regression_Housing.ipynb

---

## 📌 Notes
Ensure Housing.csv is located inside the data/ directory.
This project fulfills the core steps required for understanding linear regression in machine learning.

---