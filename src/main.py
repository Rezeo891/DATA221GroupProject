from logistic_regression_model import run_logistic_regression
from decision_tree_model import run_decision_tree
from random_forest_model import run_random_forest
from svm_model import run_svm

def main():
    all_results = {
        "Logistic Regression": run_logistic_regression(),
        "Decision Tree": run_decision_tree(),
        "Random Forest": run_random_forest(),
        "SVM": run_svm()
    }

    print("\n=== Final Model Comparison Summary ===")
    for model_name, results in all_results.items():
        print(
            f"{model_name}: "
            f"Accuracy={results['Accuracy']:.4f}, "
            f"Precision={results['Precision']:.4f}, "
            f"Recall={results['Recall']:.4f}, "
            f"F1-score={results['F1-score']:.4f}, "
            f"ROC-AUC={results['ROC-AUC']:.4f}"
        )

if __name__ == "__main__":
    main()