# Outout of the Model.

#### I used built in dataset for this problem.

============================================================
CREDITWORTHINESS PREDICTION - CODEALPHA ML TASK
============================================================

[1] Loading German Credit dataset...
Dataset shape: (1000, 21)

Features (20):
['checking_status', 'duration', 'credit_history', 'purpose', 'credit_amount', 'savings_status', 'employment', 'installment_commitment', 'personal_status', 'other_parties', 'residence_since', 'property_magnitude', 'age', 'other_payment_plans', 'housing', 'existing_credits', 'job', 'num_dependents', 'own_telephone', 'foreign_worker']

Class distribution:
target
1    700
0    300
Name: count, dtype: int64

First 5 rows:
  checking_status  duration                  credit_history              purpose  credit_amount    savings_status  ... existing_credits                 job num_dependents own_telephone  foreign_worker target
0              <0         6  critical/other existing credit             radio/tv           1169  no known savings  ...                2             skilled              1           yes             yes      1
1        0<=X<200        48                   existing paid             radio/tv           5951              <100  ...                1             skilled              1          none             yes      0
2     no checking        12  critical/other existing credit            education           2096              <100  ...                1  unskilled resident              2          none             yes      1
3              <0        42                   existing paid  furniture/equipment           7882              <100  ...                1             skilled              2          none             yes      1
4              <0        24              delayed previously              new car           4870              <100  ...                2             skilled              2          none             yes      0

[5 rows x 21 columns]

[2] Feature Engineering & Preprocessing...
Numeric features (7): ['duration', 'credit_amount', 'installment_commitment', 'residence_since', 'age', 'existing_credits', 'num_dependents']
Categorical features (13): ['checking_status', 'credit_history', 'purpose', 'savings_status', 'employment', 'personal_status', 'other_parties', 'property_magnitude', 'other_payment_plans', 'housing', 'job', 'own_telephone', 'foreign_worker']
No missing values found.
Creating derived features...
  - Created 'monthly_payment' (credit_amount / duration)
  - Created 'age_group' binned feature
  - Created 'employment_score' normalized
  - Created 'amount_per_duration' risk metric

Final feature matrix: (1000, 24)

[3] Train/Test Split & Scaling...
Train: (750, 24), Test: (250, 24)

[4] Training & Evaluating Models...
============================================================

──────────────────────────────────────────────────
  Logistic Regression
──────────────────────────────────────────────────
  Accuracy : 0.7120
  Precision: 0.7537
  Recall   : 0.8743
  F1-Score : 0.8095
  ROC-AUC  : 0.7579

  Classification Report:
                precision    recall  f1-score   support

     Bad (0)       0.53      0.33      0.41        75
    Good (1)       0.75      0.87      0.81       175

    accuracy                           0.71       250
   macro avg       0.64      0.60      0.61       250
weighted avg       0.69      0.71      0.69       250

  CV ROC-AUC (5-fold): 0.7195 (+/- 0.0478)

──────────────────────────────────────────────────
  Decision Tree
──────────────────────────────────────────────────
  Accuracy : 0.7040
  Precision: 0.7953
  Recall   : 0.7771
  F1-Score : 0.7861
  ROC-AUC  : 0.6642

  Classification Report:
                precision    recall  f1-score   support

     Bad (0)       0.51      0.53      0.52        75
    Good (1)       0.80      0.78      0.79       175

    accuracy                           0.70       250
   macro avg       0.65      0.66      0.65       250
weighted avg       0.71      0.70      0.71       250

  CV ROC-AUC (5-fold): 0.6647 (+/- 0.0355)

──────────────────────────────────────────────────
  Random Forest
──────────────────────────────────────────────────
  Accuracy : 0.7680
  Precision: 0.7773
  Recall   : 0.9371
  F1-Score : 0.8497
  ROC-AUC  : 0.7843

  Classification Report:
                precision    recall  f1-score   support

     Bad (0)       0.72      0.37      0.49        75
    Good (1)       0.78      0.94      0.85       175

    accuracy                           0.77       250
   macro avg       0.75      0.66      0.67       250
weighted avg       0.76      0.77      0.74       250

  CV ROC-AUC (5-fold): 0.7762 (+/- 0.0453)

============================================================
  MODEL COMPARISON SUMMARY
============================================================
              Model  Accuracy  Precision  Recall  F1-Score  ROC-AUC
Logistic Regression     0.712     0.7537  0.8743    0.8095   0.7579
      Decision Tree     0.704     0.7953  0.7771    0.7861   0.6642
      Random Forest     0.768     0.7773  0.9371    0.8497   0.7843

──────────────────────────────────────────────────
  Top 10 Feature Importances (Random Forest)
──────────────────────────────────────────────────
  1. checking_status                0.1102
  2. monthly_payment                0.1018
  3. credit_amount                  0.1002
  4. amount_per_duration            0.0930
  5. duration                       0.0787
  6. age                            0.0739
  7. purpose                        0.0646
  8. credit_history                 0.0474
  9. savings_status                 0.0309
  10. property_magnitude             0.0287

============================================================
  TASK COMPLETED SUCCESSFULLY
============================================================
