
# SVM Classification Project - Breast Cancer Dataset

## Objective
Classify breast cancer as benign or malignant using Support Vector Machines (SVM) with both linear and RBF kernels.

## Dataset
The dataset is located in `data/breast-cancer.csv`. It includes features extracted from digitized images of breast mass (mean, se, worst radius, texture, etc.)

## Features
- Train/Test SVM using linear and RBF kernels
- Hyperparameter tuning using GridSearchCV
- 2D PCA projection for decision boundary visualization
- Cross-validation scores

## Requirements
Install dependencies using:

```bash
pip install -r requirements.txt
```

## Run Instructions

### Run the Python script
```bash
python svm-classifier.py
```
## Output
- Accuracy, precision, recall for each kernel
- Best parameters from GridSearch
- Decision boundary plotted using PCA
