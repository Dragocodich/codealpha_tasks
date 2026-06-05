import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, precision_score, recall_score, f1_score, accuracy_score
)

print("=" * 60)
print("CREDITWORTHINESS PREDICTION - CODEALPHA ML TASK")
print("=" * 60)


print("\n[1] Loading German Credit dataset...")
from sklearn.datasets import fetch_openml
data = fetch_openml("credit-g", version=1, parser="auto")
df = pd.DataFrame(data.data, columns=data.feature_names)
df["target"] = data.target.map({"good": 1, "bad": 0}).astype(int)

print(f"Dataset shape: {df.shape}")
print(f"\nFeatures ({len(data.feature_names)}):")
print(data.feature_names)
print(f"\nClass distribution:\n{df['target'].value_counts()}")
print(f"\nFirst 5 rows:")
print(df.head())


print("\n[2] Feature Engineering & Preprocessing...")


numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
numeric_cols.remove("target")
categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

print(f"Numeric features ({len(numeric_cols)}): {numeric_cols}")
print(f"Categorical features ({len(categorical_cols)}): {categorical_cols}")


if df.isnull().sum().sum() > 0:
    print(f"Missing values found, filling...")
    for col in numeric_cols:
        df[col].fillna(df[col].median(), inplace=True)
    for col in categorical_cols:
        df[col].fillna(df[col].mode()[0], inplace=True)
else:
    print("No missing values found.")


le_dict = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    le_dict[col] = le


print("Creating derived features...")


if "credit_amount" in df.columns and "duration" in df.columns:
    df["monthly_payment"] = df["credit_amount"] / df["duration"]
    print("  - Created 'monthly_payment' (credit_amount / duration)")


if "age" in df.columns:
    df["age_group"] = pd.cut(df["age"], bins=[0, 25, 35, 50, 100], labels=[0, 1, 2, 3]).astype(int)
    print("  - Created 'age_group' binned feature")


if "employment" in df.columns:
    df["employment_score"] = df["employment"] / df["employment"].max()
    print("  - Created 'employment_score' normalized")


if "credit_amount" in df.columns and "duration" in df.columns:
    df["amount_per_duration"] = df["credit_amount"] / (df["duration"] + 1)
    print("  - Created 'amount_per_duration' risk metric")


feature_cols = [c for c in df.columns if c != "target"]
X = df[feature_cols].values
y = df["target"].values

print(f"\nFinal feature matrix: {X.shape}")


print("\n[3] Train/Test Split & Scaling...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)
print(f"Train: {X_train.shape}, Test: {X_test.shape}")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


print("\n[4] Training & Evaluating Models...")
print("=" * 60)

models = {
    "Logistic Regression": LogisticRegression(max_iter=2000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(max_depth=8, min_samples_split=10, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_split=5, random_state=42),
}

results = []

for name, model in models.items():
    print(f"\n{'─' * 50}")
    print(f"  {name}")
    print(f"{'─' * 50}")

    if name == "Decision Tree":
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)

    print(f"  Accuracy : {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall   : {rec:.4f}")
    print(f"  F1-Score : {f1:.4f}")
    print(f"  ROC-AUC  : {roc_auc:.4f}")
    print(f"\n  Classification Report:")
    print(f"  {classification_report(y_test, y_pred, target_names=['Bad (0)', 'Good (1)'])}")

    results.append({
        "Model": name,
        "Accuracy": round(acc, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "F1-Score": round(f1, 4),
        "ROC-AUC": round(roc_auc, 4),
    })

    # Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    if name == "Decision Tree":
        cv_scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")
    else:
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring="roc_auc")
    print(f"  CV ROC-AUC (5-fold): {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")


print(f"\n{'=' * 60}")
print("  MODEL COMPARISON SUMMARY")
print(f"{'=' * 60}")
results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))


print(f"\n{'─' * 50}")
print("  Top 10 Feature Importances (Random Forest)")
print(f"{'─' * 50}")
importances = models["Random Forest"].feature_importances_
indices = np.argsort(importances)[::-1][:10]
for i, idx in enumerate(indices):
    print(f"  {i+1}. {feature_cols[idx]:30s} {importances[idx]:.4f}")

print(f"\n{'=' * 60}")
print("  TASK COMPLETED SUCCESSFULLY")
print(f"{'=' * 60}")
