from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from src.utils import load_data
import graphviz

def train_decision_tree():
    X_train, X_test, y_train, y_test = load_data()
    clf = DecisionTreeClassifier(max_depth=3, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print("Decision Tree Accuracy:", accuracy_score(y_test, y_pred))
    scores = cross_val_score(clf, X_train, y_train, cv=5)
    print("Cross-validation accuracy:", scores.mean())

    export_graphviz(clf, out_file="visuals/decision_tree.dot", 
                    feature_names=X_train.columns, 
                    class_names=["No Disease", "Disease"],
                    filled=True, rounded=True, special_characters=True)