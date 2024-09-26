# Fraud Detection Project for Fidelity Bank PLC

## Project Overview
This project involves creating a machine-learning model to detect fraudulent transactions in the bank's credit card data. The dataset utilized comes from Kaggle and includes various credit card transaction-related attributes. The purpose is to detect possibly fraudulent transactions using classification methods like Logistic Regression and XGBoost.

## Libraries Used
- Python 3.x
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost
- Jupyter Notebook

## Dataset
This project's dataset is a cleaned version of credit card transaction records, complete with a binary target variable indicating whether a transaction is fraudulent indicates whether a transaction is fraudulent or not. The dataset includes transaction amount, time, and various anonymized features.

## Installation Instructions
To run this project, you must install Python 3.x on your machine. 
Steps to set up the environment:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/mkazu1234/BAN6800_Assignments
   cd MOD4_BANfraud_DetectionModel

2. Install necessary packages 
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost

## Usage
1. Run the Jupyter Notebook: Open a terminal and navigate to the project directory. 
Start Jupyter Notebook:
```bash
jupyter notebook

2. Open the Notebook: Open the notebook file (MOD4fraud_detectModel.ipynb) in the browser.

3. Execute the cells: Run each cell in the notebook sequentially to perform data analysis, model training, and evaluation.

## Evaluation Metrics
The performance of the models is evaluated using the following metrics:

Accuracy: The proportion of correctly predicted instances.
ROC-AUC Score: A metric that reflects the model's ability to discriminate between classes.
Confusion Matrix: A table that summarizes the performance of the classification algorithm.
Classification Report: Includes precision, recall, and F1-score for each class.

## Results
After training and evaluating the models, the following results were obtained:
Logistic Regression Accuracy: 0.9991
Logistic Regression ROC-AUC Score: 0.9549
XGBoost Accuracy: 0.9995
XGBoost ROC-AUC Score: 0.9709
These metrics indicate how well the models perform in detecting fraudulent transactions in the dataset.

## Summary Report
Total Transactions: 283726
Fraudulent Transactions: 473
Non-Fraudulent Transactions: 283253

## Interpretation of Results

## Model Performance

Logistic Regression
Accuracy: 0.9991: The model correctly classifies 99.91% of transactions, indicating excellent overall performance.
ROC-AUC Score: 0.9549: This score suggests strong capability in distinguishing between fraudulent and non-fraudulent transactions, with a good balance between sensitivity and specificity.

XGBoost
Accuracy: 0.9995: The XGBoost model performs slightly better, achieving an accuracy of 99.95%.
ROC-AUC Score: 0.9709: This score indicates even better discrimination ability compared to Logistic Regression, meaning it is more effective at predicting fraud.

Summary of Transactions
Total Transactions: 283,726
Fraudulent Transactions: 473 (0.17%)
Non-Fraudulent Transactions: 283,253 (99.83%)

Conclusion
Despite the considerable class imbalance, with fraudulent transactions accounting for only 0.17% of the dataset, both models achieve outstanding accuracy and ROC-AUC scores, suggesting their ability to detect fraudulent transactions. XGBoost significantly surpasses Logistic Regression in both metrics, making it a good contender for this task.