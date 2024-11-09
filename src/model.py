# src/model.py
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns


def train_model(X_train, y_train):
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    conf_mat_decision_tree = confusion_matrix(y_test, predictions)

    print("Reporte de Clasificación")
    class_report_decision_tree = classification_report(y_test, predictions, zero_division=0)
    print(class_report_decision_tree)

    print("Matriz de Confusión:")
    print(conf_mat_decision_tree)
    print("\n")

    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_mat_decision_tree, annot=True, fmt='d', cmap='Blues')
    plt.title('Matriz de Confusión')
    plt.ylabel('Actual')
    plt.xlabel('Predicción')
    plt.show()
