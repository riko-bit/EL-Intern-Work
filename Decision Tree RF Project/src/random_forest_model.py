from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from src.utils import load_data
import matplotlib.pyplot as plt
import pandas as pd

def train_random_forest():
    X_train, X_test, y_train, y_test = load_data()
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print("Random Forest Accuracy:", accuracy_score(y_test, y_pred))
    scores = cross_val_score(clf, X_train, y_train, cv=5)
    print("Cross-validation accuracy:", scores.mean())

    importances = pd.Series(clf.feature_importances_, index=X_train.columns)
    importances.sort_values(ascending=True).plot(kind='barh')
    plt.title("Feature Importances")
    plt.tight_layout()
    plt.savefig("outputs/feature_importance.png")