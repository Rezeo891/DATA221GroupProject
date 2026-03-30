import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    roc_auc_score,
    confusion_matrix
)

# Load Dataset
dataset = pd.read_csv("../data/water_potability.csv")

# Separate Features and Target
features = dataset.drop(columns=["Potability"])
target = dataset["Potability"]

# Train Test Split
features_train, features_test, target_train, target_test = train_test_split(
    features,
    target,
    test_size=0.2,
    random_state=42,
    stratify=target
)

# Median Imputation + Scaling + SVM
pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("svm", SVC(probability=True, class_weight="balanced", random_state=42))
])

# Hyperparameter Grid
param_grid = {
    "svm__C": [0.1, 1, 10, 50],
    "svm__gamma": ["scale", 0.1, 0.01],
    "svm__kernel": ["rbf"]
}

# Grid Search to optimize F1
grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring="f1",
    n_jobs=-1
)

grid_search.fit(features_train, target_train)
best_model = grid_search.best_estimator_

# Probabilities for threshold
predicted_probabilities = best_model.predict_proba(features_test)[:, 1]

# Tune threshold for best F1
best_threshold = 0.5
best_f1_score_value = 0

for threshold in np.arange(0.1, 0.9, 0.01):
    predicted_classes = (predicted_probabilities >= threshold).astype(int)
    current_f1_score = f1_score(target_test, predicted_classes)

    if current_f1_score > best_f1_score_value:
        best_f1_score_value = current_f1_score
        best_threshold = threshold

# Final predictions using threshold
final_predicted_labels = (predicted_probabilities >= best_threshold).astype(int)

# Evaluation Metrics
model_accuracy = accuracy_score(target_test, final_predicted_labels)
model_precision = precision_score(target_test, final_predicted_labels)
model_recall = recall_score(target_test, final_predicted_labels)
model_f1_score = f1_score(target_test, final_predicted_labels)
model_roc_auc = roc_auc_score(target_test, predicted_probabilities)

# Print Statements
print("Support Vector Machine Performance")
print(f"Accuracy: {model_accuracy:.4f}")
print(f"Precision: {model_precision:.4f}")
print(f"Recall: {model_recall:.4f}")
print(f"F1 Score: {model_f1_score:.4f}")
print(f"ROC-AUC: {model_roc_auc:.4f}")

# Confusion Matrix
confusion_matrix_result = confusion_matrix(target_test, final_predicted_labels)
print("Confusion Matrix:")
print(confusion_matrix_result)

# ROC Curve Plotting
false_positive_rate, true_positive_rate, threshold_values = roc_curve(
    target_test,
    predicted_probabilities
)

plt.figure()
plt.plot(false_positive_rate, true_positive_rate, label=f"SVM (AUC = {model_roc_auc:.3f})")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Support Vector Machine")
plt.legend(loc="lower right")
plt.grid()
plt.show()
