from sklearn.tree import DecisionTreeClassifier
from preprocessing import load_and_preprocess
from utils import evaluate_model, print_results


def run_decision_tree():
    X_train, X_test, y_train, y_test = load_and_preprocess()

    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)

    results = evaluate_model(model, X_test, y_test)
    print_results("Decision Tree", results)

    return results


if name == "main":
    run_decision_tree()
