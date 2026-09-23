# Financial Loan Risk — Loan Approval Prediction

## Project Overview

This project uses machine learning to predict whether a loan application will be approved based on an applicant's financial, credit, and loan characteristics.

The project was developed as a decision-support tool for loan officers, with particular attention to the different financial costs of false-positive and false-negative predictions.

**False negative:** $8,000 estimated cost
**False positive:** $50,000 estimated cost

Because false positives are substantially more costly, the project uses a business-cost-based classification threshold rather than relying solely on the default 0.50 threshold.

---

## Objectives

* Explore factors associated with loan approval.
* Prepare and preprocess financial and categorical data.
* Compare multiple classification models.
* Tune the selected model using cross-validation.
* Optimize the prediction threshold based on business costs.
* Evaluate the final model on an unseen test set.

---

## Exploratory Data Analysis

The analysis examined applicant financial and credit characteristics, including:

* Credit Score
* Monthly Income
* Debt-to-Income Ratio
* Length of Credit History
* Loan Amount
* Interest Rate
* Employment and education characteristics

Approved applicants generally showed higher income and credit scores and lower debt-to-income ratios.

However, substantial overlap existed between approved and denied applicants, indicating that loan approval could not be reliably determined from a single variable.

---

## Data Preparation

The dataset contains **20,000 loan applications and 35 features**.

The project included:

* Missing-value imputation
* Numerical feature scaling
* Ordinal encoding
* One-hot encoding of categorical variables
* Stratified train/validation/test splitting
* Cross-validation

Potential data leakage was also investigated. `RiskScore` was excluded because it appeared to contain information closely related to the historical approval decision.

---

## Modeling

Three classification approaches were compared:

* Logistic Regression
* Random Forest
* Histogram Gradient Boosting

After cross-validation and hyperparameter tuning, **Logistic Regression** achieved the highest validation ROC-AUC.

The final operating threshold was selected using the validation set to minimize the project's assumed business cost of classification errors.

---

## Final Results

The final model was evaluated on a previously unseen test set of 4,000 applications.

| Metric               | Result |
| -------------------- | -----: |
| Accuracy             |   0.95 |
| Precision — Denied   |   0.94 |
| Recall — Denied      |   0.99 |
| Precision — Approved |   0.98 |
| Recall — Approved    |   0.81 |
| F1 — Approved        |   0.89 |

Under the project's assumed error costs, the tuned model produced an estimated test-set cost of **$2.264M**, compared with **$7.648M** for the "approve no one" baseline and **$152.2M** for the "approve everyone" baseline.

---

## Key Takeaways

* Financial capacity and credit-related variables were important predictors of approval.
* Model evaluation should consider the business cost of prediction errors, not accuracy alone.
* Data leakage needs to be considered carefully when using historical approval decisions.
* The model is best treated as a **decision-support tool**, with human review retained for individual applications.

---

## Technologies

**Python | Pandas | NumPy | Scikit-learn | Matplotlib | Seaborn | Jupyter Notebook**

### Machine Learning

Logistic Regression • Random Forest • Histogram Gradient Boosting • Cross-validation • Hyperparameter tuning • Classification metrics • Threshold optimization
