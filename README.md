# Credit Risk Prediction Analysis
## Overview

This notebook performs an analysis and comparison of two machine learning models—Logistic Regression and XGBoost—for predicting loan default risk. The goal is to assess the effectiveness of both models in predicting whether a borrower will default on a loan, based on a set of features such as age, income, credit history, and more. The models are trained and evaluated on a publicly available credit risk dataset.
Dataset

The dataset used in this analysis includes the following variables:
Variable Name	Description
person_age	Age of the borrower in years.
person_income	Annual income of the borrower (in dollars).
person_home_ownership	Type of home ownership (e.g., RENT, OWN, MORTGAGE).
person_emp_length	Employment length of the borrower (in years).
loan_intent	Purpose of the loan (e.g., PERSONAL, EDUCATION, MEDICAL).
loan_grade	Credit grade of the loan, typically assigned by a lender (e.g., A, B, C).
loan_amnt	Amount of the loan requested (in dollars).
loan_int_rate	Interest rate for the loan (as a percentage).
loan_status	Target variable: indicates whether the borrower defaulted (1) or not (0).
loan_percent_income	Ratio of the loan amount to the borrower's annual income.
cb_person_default_on_file	Whether the borrower has a default on file, encoded as Y (Yes) or N (No).
cb_person_cred_hist_length	Length of the borrower’s credit history (in years).

## Analysis and Models

    Preprocessing:
        Missing values are imputed using the median strategy.
        Categorical variables are encoded using Label Encoding.
        Data is split into training and testing sets (80/20 split).
        Features are standardized using StandardScaler.

    Models:
        Logistic Regression: A simple linear model, with a class-weight adjustment to account for class imbalance.
        XGBoost: A gradient boosting machine (GBM) model with scale adjustment for positive weights to handle class imbalance.

    Model Evaluation:
        Both models are evaluated using accuracy, confusion matrices, AUC-ROC, and classification reports.
        Hyperparameter tuning was conducted via GridSearchCV to optimize model performance.

## Key Findings

    Model Performance: The XGBoost model outperforms the Logistic Regression model, which is expected due to XGBoost's ability to handle complex relationships and interactions between features.
    Important Features: The most important predictor of loan default is loan_grade, which is a pre-calculated and complex variable. This variable is key in determining a borrower's creditworthiness.
    Credit History: Having a default on file (cb_person_default_on_file) is an important feature in predicting loan default risk, but it is not 100% accurate. Some individuals with a past default on file have successfully repaid their loans, highlighting the possibility of recovery.
    Clean Credit Histories: Even individuals with no default history may still default on their loans, underscoring the complexity of credit risk prediction.

## Conclusion

This analysis demonstrates that a well-trained machine learning model, such as XGBoost, can be effective in predicting loan default risk. While past credit history is an important factor, other variables such as loan grade, income, and employment history also play significant roles in determining the likelihood of loan default.
How to Use This Notebook

    Clone or download the notebook.
    Make sure you have the required libraries installed:
        pandas
        numpy
        scikit-learn
        xgboost
        matplotlib
        seaborn
    Load the dataset and run the cells sequentially.
    The notebook will output model evaluations, performance metrics, and visualizations.

## Dependencies

The following Python packages are required to run this notebook:
pip install pandas numpy scikit-learn xgboost matplotlib seaborn

License

This project is licensed under the MIT License - see the LICENSE file for details.
