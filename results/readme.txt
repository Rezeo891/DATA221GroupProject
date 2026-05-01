DATA 221 Course Project – Part A: Research Proposal

Water Potability Prediction: A Comparative Study of Machine Learning Models for Safe Drinking Water Classification
Group Members: Rudraksh Walia, Lucas Zhang, Yiming（Alan）Wu
Affiliation: Department of Data Science, University of Calgary
---
1. Abstract
Access to safe drinking water remains a critical global health challenge, as contaminated water causes numerous diseases and poses serious health risks. This project will address the problem of binary classification of water potability based on physicochemical properties. Using the publicly available “Water Potability” dataset from Kaggle, which contains 9 features and 3,276 water samples with missing values, the task will be framed as a classification problem. The project will compare four machine learning models: Support Vector Machine (SVM), Logistic Regression, Decision Tree, and Random Forest. After applying median imputation for missing values and feature scaling with StandardScaler, an 80/20 stratified train-test split will be performed. Model performance will be evaluated using accuracy, precision, recall, F1-score, and ROC-AUC. The expected outcome is to identify which model provides the most reliable and balanced prediction of water potability, with particular attention to minimizing false safety predictions (classifying unsafe water as potable). The Random Forest is hypothesized to achieve the best overall performance due to its ensemble nature and robustness to overfitting.
---
2. Introduction
Paragraph 1 – Problem and Importance: Contaminated drinking water is a major cause of diseases such as cholera, typhoid, and diarrhea, leading to millions of preventable deaths annually, especially in developing regions. Water quality depends on multiple chemical and physical factors, including pH, hardness, solids, chloramines, sulfate, conductivity, organic carbon, trihalomethanes, and turbidity. Manual laboratory testing is time-consuming, expensive, and often not scalable for continuous monitoring of water sources. The ability to predict water potability quickly and accurately using machine learning can enable early detection of unsafe water, support governments and organizations in making informed decisions about water treatment and distribution, and ultimately reduce public health risks. A data-driven, scalable approach to water quality assessment is therefore of high societal and economic importance.

Paragraph 2 – Previous Work and Research Gap: Several studies have applied machine learning to water quality prediction. Researchers have used logistic regression, decision trees, random forests, and support vector machines on similar datasets, often achieving moderate accuracy between 60% and 70%. For instance, previous work on the Kaggle “Water Potability” dataset reported baseline results using simple classifiers, but many existing studies either did not address class imbalance properly, used different train-test splits, or focused on only one or two models without rigorous comparison. Moreover, previous analyses often did not emphasize the critical trade-off between false positives (labeling unsafe water as safe) and false negatives (labeling safe water as unsafe). The research gap is a systematic, fair comparison of multiple models on the same data partition with explicit handling of missing values and scaling, coupled with a clear discussion of which model best balances safety and reliability in real-world deployment.

Paragraph 3 – Your Contribution: This project will develop and compare four machine learning models – SVM, Logistic Regression, Decision Tree, and Random Forest – on the same standardized train-test split to ensure fair evaluation. We will implement median imputation for missing values, apply StandardScaler, and use stratification to preserve class proportions. Hyperparameter tuning will be performed using grid search, and class imbalance will be addressed via class weighting. The primary contribution is a rigorous, side-by-side comparison using multiple classification metrics (accuracy, precision, recall, F1-score, ROC-AUC) and confusion matrix analysis, with a specific focus on identifying which model minimizes the risk of incorrectly labeling unsafe water as potable. The results will provide practical guidance for deploying machine learning in water quality monitoring systems.
---
3. Method
Dataset Description

The dataset, titled “Water Potability,” will be obtained from Kaggle (https://www.kaggle.com/datasets/adityakadiwal/water-potability). It contains 3,276 water samples, each with 9 physicochemical features and one binary target variable (Potability), where 0 indicates “not potable” (unsafe) and 1 indicates “potable” (safe). The features are: pH, Hardness, Solids, Chloramines, Sulfate, Conductivity, Organic_carbon, Trihalomethanes, and Turbidity. The dataset has missing values in several features. The class distribution is moderately imbalanced: approximately 61% of samples are labeled not potable and 39% potable.
Data Splitting Plan
We will split the dataset into training and testing sets using an 80/20 ratio. To ensure that both classes are represented proportionally in both splits, we will use stratified random sampling. The same fixed random seed will be used to guarantee reproducibility of the split across all models. The training set will be used for model training and hyperparameter tuning (via cross-validation), while the testing set will be reserved exclusively for final evaluation. We will take care to avoid any data leakage by performing all preprocessing (imputation and scaling) using parameters estimated only from the training data, then applying those same parameters to the test set.
Preprocessing Plan
· Handling missing values: We will use median imputation for all numeric features that contain missing values. The median will be computed from the training set only and then applied to both training and test sets.
· Feature scaling: All features will be standardized using StandardScaler (zero mean, unit variance) to ensure that models sensitive to feature magnitudes (e.g., SVM, Logistic Regression) perform optimally. Scaling parameters will be fit on the training data and transformed on the test data.
· Class imbalance: For models that support it (SVM, Logistic Regression, Random Forest), we will apply the class_weight="balanced" parameter to adjust weights inversely proportional to class frequencies. For the Decision Tree, we will also explore setting class weights. Additionally, we will compare results with and without weighting to assess its impact.
Planned Models
We will implement and compare exactly four machine learning models:

1. Logistic Regression (baseline linear model): A simple linear classifier that will serve as a reference point. We will set max_iter=1000 to ensure convergence.
2. Support Vector Machine (SVM) with RBF kernel: A non-linear model capable of capturing complex decision boundaries. Hyperparameters to tune include gamma and C using GridSearchCV.
3. Decision Tree: A non-linear, tree-based model that recursively splits features based on impurity reduction. We will limit maximum depth to prevent overfitting and tune hyperparameters such as max_depth and min_samples_split.
4. Random Forest: An ensemble of decision trees that averages predictions to reduce variance and overfitting. We will tune n_estimators, max_depth, and min_samples_split.

Each model will be trained and evaluated using the exact same training and test splits. Each team member will be responsible for implementing and running at least one model: one member for Logistic Regression, one for SVM, one for Decision Tree, and one for Random Forest. All code will be developed in individual GitHub repositories and then merged into a team repository.
Evaluation Metrics
Since this is a binary classification task with class imbalance and asymmetric consequences (false safety is more dangerous than false alarm), we will use the following metrics:

· Accuracy: Overall proportion of correct predictions.
· Precision: Among samples predicted as potable, how many are truly potable.
· Recall (Sensitivity): Among truly potable samples, how many were correctly identified.
· F1-score: Harmonic mean of precision and recall.
· ROC-AUC: Area under the Receiver Operating Characteristic curve, measuring the model’s ability to discriminate between classes.

Confusion matrices will also be generated to visualize true positives, false positives, true negatives, and false negatives.
Success Criteria

“Better performance” will be defined primarily by the model’s ability to minimize false positives (predicting unsafe water as potable) while maintaining reasonable recall for potable water. Specifically, we will consider a model successful if it achieves:

· An F1-score of at least 0.60 on the test set, and
· A false positive rate (FPR) lower than 0.30, with preference for models that have the lowest FPR even at the cost of some recall, because incorrectly labeling unsafe water as safe poses direct health risks.

Among the four models, the one with the highest F1-score and the lowest false positive rate will be recommended as the best model for practical deployment.
---
4. References
[1] Kaggle. (n.d.). Water Potability Dataset. Retrieved from https://www.kaggle.com/datasets/adityakadiwal/water-potability
[2] Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. Journal of Machine Learning Research, 12, 2825–2830.
[3] Kuhn, M., & Johnson, K. (2013). Applied Predictive Modeling. Springer.
[4] Géron, A. (2019). Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow (2nd ed.). O’Reilly Media.
[5] Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining.
(Note: Reference [5] is included for potential future extension but will not be used in the current four-model comparison.)
