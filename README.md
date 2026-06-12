Cancer Detection using Machine Learning
A machine learning project to predict the presence of cancer based on White Blood Cell (WBC) test reports. Three classification algorithms are compared — Decision Tree, Logistic Regression, and KNN — with a fully interactive Streamlit dashboard.

##  Problem Statement

From the given `wbc.csv` dataset containing WBC test reports of patients, predict whether cancer is **present (Malignant)** or **absent (Benign)**

##  Dataset Info

| Property | Details |
|---|---|
| Source | WBC (White Blood Cell) CSV |
| Records | 569 patients |
| Features | 30 numeric WBC test features |
| Target | `diagnosis` → M (Malignant = 1) / B (Benign = 0) |
| Class Distribution | 357 Benign · 212 Malignant 

Tech Stack

Language: Python
Libraries: Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, Streamlit
Environment: Google Colab (Notebook) · PyCharm (Dashboard)

ML Models Used
1. Decision Tree Classifier

Used Gini impurity as splitting criterion
Applied Hit and Trial method using cross-validation to find optimal max_depth
Best depth found: 5

2. Logistic Regression

Applied directly on unscaled data
Used default hyperparameters

3. K-Nearest Neighbors (KNN)

Applied cross-validation to find the best value of k
Data scaled using StandardScaler before training



