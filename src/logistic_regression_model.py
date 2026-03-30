from sklearn.linear_model import LogisticRegression
from preprocessing import load_and_preprocess
from utils import evaluate_model, print_results


def run_logistic_regression():
    X_train, X_test, y_train, y_test = load_and_preprocess()

    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train, y_train)

    results = evaluate_model(model, X_test, y_test)
    print_results("Logistic Regression", results)

    return results


if __name__ == "__main__":
    run_logistic_regression()