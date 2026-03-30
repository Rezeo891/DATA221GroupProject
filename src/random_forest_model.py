from sklearn.ensemble import RandomForestClassifier
from preprocessing import load_and_preprocess
from utils import evaluate_model, print_results


def run_random_forest():
    X_train, X_test, y_train, y_test = load_and_preprocess()

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
    model.fit(X_train, y_train)

    results = evaluate_model(model, X_test, y_test)
    print_results("Random Forest", results)

    return results


if name == "main":
    run_random_forest()
