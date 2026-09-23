# Financial Loan Risk — Loan Approval Prediction

## Project Overview

This project develops a machine-learning classification model to support loan approval decisions for a fictional FinTech company.

The goal is not simply to maximize prediction accuracy, but to build a model that balances predictive performance with the financial consequences of incorrect loan decisions.

The business assigns different costs to the two types of classification errors:

* **False negative:** Denying a creditworthy applicant — estimated cost of **$8,000**
* **False positive:** Approving a loan that later defaults — estimated cost of **$50,000**

Because a false positive is substantially more costly, the project uses a **business-cost-based decision threshold** rather than relying on the default 0.50 classification threshold.

The final model is designed as a **decision-support risk score for loan officers**, rather than a fully automated approval system.

---

## Business Problem

The existing loan approval process relies on manual review by loan officers. This can be slow, inconsistent across reviewers, and difficult to scale as application volume increases.

The proposed machine-learning solution aims to provide loan officers with a consistent risk signal that can:

* Support faster application review
* Identify patterns associated with loan approval
* Reduce costly approval errors
* Retain human oversight for individual cases

---

## Dataset

The dataset contains **20,000 loan applications** and **35 columns**, including applicant demographics, financial characteristics, credit history, loan characteristics, and the historical approval decision.

The target variable is:

* `LoanApproved` — binary outcome indicating whether a loan application was approved.

The dataset contains approximately **24% approved applications**, making class imbalance an important consideration during modeling and evaluation.

### Missing Data

Missing values were identified in:

* `EducationLevel` — 901 missing values
* `MaritalStatus` — 1,331 missing values
* `SavingsAccountBalance` — 572 missing values

These were handled within the modeling pipeline using appropriate imputation strategies.

---

## Objectives

The project aims to:

1. Explore the characteristics associated with loan approval.
2. Identify important financial and credit-related predictors.
3. Build and compare multiple classification models.
4. Tune the selected models using cross-validation.
5. Optimize the classification threshold according to the business cost of prediction errors.
6. Evaluate the final model on a previously unseen test set.
7. Examine model performance across credit-score segments.
8. Investigate potential data leakage and feature-timing concerns.

---

## Exploratory Data Analysis

Exploratory analysis examined numerical and categorical variables in relation to loan approval.

Some of the strongest linear associations with `LoanApproved` were:

* `MonthlyIncome`
* `AnnualIncome`
* `TotalDebtToIncomeRatio`
* `InterestRate`
* `BaseInterestRate`
* `LoanAmount`

Approved applicants generally showed patterns of:

* Higher income
* Higher credit scores
* Lower debt-to-income ratios
* Longer credit histories

However, the distributions showed substantial overlap between approved and denied applicants, indicating that no single feature could reliably determine the outcome.

Categorical approval rates were also examined across variables including:

* Employment status
* Education level
* Home ownership
* Loan purpose

---

## Data Leakage Check

`RiskScore` was excluded from the model because it showed a strong correlation with the target (`r ≈ -0.77`) and appeared to encode information closely related to the historical approval decision.

Including it as a predictor would risk allowing the model to indirectly access the answer it is supposed to predict.

Loan-term variables were also investigated for possible leakage. Removing `InterestRate`, `BaseInterestRate`, and `MonthlyLoanPayment` reduced validation ROC-AUC from approximately **0.9945 to 0.9772**, a relatively small decrease. The notebook therefore recommends confirming when these fields are established in the actual loan-origination process before production use.

---

## Data Preparation

The modeling pipeline separates features into:

* **27 numerical features**
* **1 ordinal feature**
* **5 nominal categorical features**

### Numerical features

Missing numerical values were handled using median imputation, followed by standardization.

### Ordinal feature

`EducationLevel` was encoded according to its natural order:

`High School < Associate < Bachelor < Master < Doctorate`

Missing values were imputed before encoding.

### Nominal categorical features

Categorical variables were imputed using the most frequent category and encoded using one-hot encoding.

All preprocessing was implemented inside a `ColumnTransformer` and `Pipeline` so that preprocessing was fitted only on the appropriate training folds during cross-validation.

---

## Modeling

Three classification algorithms were compared:

### 1. Logistic Regression

Used as an interpretable baseline and because its coefficients can be more easily communicated to loan officers and regulators.

### 2. Random Forest

Used to capture non-linear relationships and interactions between applicant characteristics.

### 3. Histogram Gradient Boosting

Used as a powerful tree-based approach for tabular data.

The data was split into:

* **Training:** 12,000 applications
* **Validation:** 4,000 applications
* **Test:** 4,000 applications

Stratified splitting was used to preserve the approval-class distribution.

Five-fold stratified cross-validation was used during model comparison and hyperparameter tuning.

---

## Model Comparison

Cross-validation produced the following mean ROC-AUC results:

| Model                | Mean CV ROC-AUC |
| -------------------- | --------------: |
| Logistic Regression  |          0.9945 |
| HistGradientBoosting |          0.9902 |
| Random Forest        |          0.9775 |

Hyperparameter tuning was performed using `RandomizedSearchCV`.

The tuned Logistic Regression achieved the highest cross-validation ROC-AUC at approximately **0.9946**.

---

## Business-Cost Threshold Optimization

A default classification threshold of 0.50 does not account for the unequal financial consequences of false positives and false negatives.

The project therefore defined:

**Business Cost = $8,000 × False Negatives + $50,000 × False Positives**

The operating threshold was selected using the validation set by searching for the threshold that minimized expected business cost.

The final selected model was:

**Logistic Regression**

with an operating threshold of:

**0.95**

The test set was not used during model or threshold selection.

---

## Final Model Performance

The final model was evaluated once on the held-out test set after the model and operating threshold had been locked.

### Classification performance

| Metric               | Result |
| -------------------- | -----: |
| Accuracy             |   0.95 |
| Precision — Denied   |   0.94 |
| Recall — Denied      |   0.99 |
| F1 — Denied          |   0.97 |
| Precision — Approved |   0.98 |
| Recall — Approved    |   0.81 |
| F1 — Approved        |   0.89 |

The model achieved strong discrimination while prioritizing avoidance of costly false-positive approvals.

### Expected business cost

On the 4,000-application test set:

| Strategy                  | Expected Cost |
| ------------------------- | ------------: |
| Approve everyone          |       $152.2M |
| Approve no one            |       $7.648M |
| Tuned Logistic Regression |   **$2.264M** |

The tuned model therefore produced substantially lower expected test-set cost than the two simple baseline strategies under the project's assumed error costs.

---

## Feature Importance

The analysis found that financial capacity and debt-related characteristics were particularly important to the model.

Important variables included measures related to:

* Income
* Debt-to-income ratios
* Credit history
* Credit score
* Loan characteristics

These findings are consistent with the objective of assessing an applicant's financial capacity and credit risk.

---

## Segment Analysis

Model behavior was also examined across credit-score bands:

* Poor: `<580`
* Fair: `580–670`
* Good: `670–740`
* Very Good: `740+`

The test set contained relatively few observations in the higher credit-score bands, so these segment results should be interpreted cautiously.

A full fairness assessment by protected characteristics was not possible from the available data and would be required before production deployment.

---

## Business Recommendations

Based on the analysis:

1. Use the model as a **decision-support risk score** rather than a fully automated approval/denial system.
2. Keep loan officers involved in reviewing individual applications and edge cases.
3. Confirm the timing of loan-term variables with the business/data team to rule out process-related leakage.
4. Monitor model performance across relevant demographic segments where legally and ethically appropriate.
5. Retrain and monitor the model as the applicant population changes.
6. Compare the model-assisted workflow with the existing manual process before full deployment.
7. Where possible, develop future models using actual repayment outcomes rather than historical approval decisions.

---

## Limitations

The target variable represents **historical loan approval decisions**, not whether an approved loan was ultimately repaid or defaulted.

As a result, the model may reproduce patterns or biases present in historical approval decisions.

The assumed costs of $8,000 for false negatives and $50,000 for false positives are business assumptions used for the project and should be validated against actual financial outcomes.

The available data also does not provide enough information for a complete fairness audit.

---

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn
* Jupyter Notebook

### Machine Learning Techniques

* Logistic Regression
* Random Forest
* Histogram Gradient Boosting
* Cross-validation
* Randomized hyperparameter search
* ROC-AUC evaluation
* Precision, recall and F1-score
* Confusion matrix analysis
* Threshold optimization
* Permutation/feature importance
* Data leakage analysis

---

## Project Structure

```text
financial_loan_risk/
│
├── ML_Model_For_LoanApproval.ipynb
└── README.md
```

---

## Key Takeaway

This project demonstrates an end-to-end machine-learning workflow for a business classification problem, from exploratory analysis and preprocessing through model comparison, hyperparameter tuning, threshold optimization, and final evaluation.

Rather than optimizing solely for accuracy, the project incorporates the **real-world cost of different prediction errors** into the modeling process and retains human oversight as part of the proposed deployment approach.
