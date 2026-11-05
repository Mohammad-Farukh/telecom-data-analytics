
# =====================================================
# WEEK 3: TELECOM CUSTOMER CHURN PREDICTION
# =====================================================

# 1. Import Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from sklearn.inspection import permutation_importance


# 2. Load Dataset

df = pd.read_csv("Telco-Customer-Churn.csv")

print("Original Dataset Shape:", df.shape)
print("\nFirst Five Records:")
print(df.head())


# 3. Data Cleaning

# Convert TotalCharges into numeric format
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"].astype(str).str.strip(),
    errors="coerce"
)

print("\nMissing Values Before Cleaning:")
print(df.isnull().sum())

print("\nMissing TotalCharges:", df["TotalCharges"].isnull().sum())

# Fill missing TotalCharges with 0 for zero-tenure customers
df.loc[
    (df["TotalCharges"].isnull()) & (df["tenure"] == 0),
    "TotalCharges"
] = 0

# Fill any remaining missing values with median
df["TotalCharges"] = df["TotalCharges"].fillna(
    df["TotalCharges"].median()
)

print("\nMissing Values After Cleaning:")
print(df.isnull().sum().sum())


# 4. Feature Engineering

# Create tenure groups
df["TenureGroup"] = pd.cut(
    df["tenure"],
    bins=[-1, 12, 24, 48, 72],
    labels=["New", "Short", "Medium", "Long"]
)

# Count subscribed services
service_columns = [
    "MultipleLines",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies"
]

df["TotalServices"] = (
    df[service_columns] == "Yes"
).sum(axis=1)

# Calculate average monthly spending
df["AverageMonthlySpend"] = np.where(
    df["tenure"] > 0,
    df["TotalCharges"] / df["tenure"],
    df["MonthlyCharges"]
)

# Create family indicator
df["HasFamily"] = (
    (df["Partner"] == "Yes") |
    (df["Dependents"] == "Yes")
).astype(int)

print("\nEngineered Features:")
print(
    df[
        [
            "TenureGroup",
            "TotalServices",
            "AverageMonthlySpend",
            "HasFamily"
        ]
    ].head()
)


# 5. Prepare Features and Target

X = df.drop(
    columns=["customerID", "Churn"]
)

y = df["Churn"].map({
    "No": 0,
    "Yes": 1
})

# Identify numerical and categorical columns
numeric_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()

print("\nNumerical Features:", numeric_features)
print("\nCategorical Features:", categorical_features)


# 6. Data Preprocessing

numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)


# 7. Create Logistic Regression Model

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=42
            )
        )
    ]
)


# 8. Split Dataset into Training and Testing

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining Records:", X_train.shape[0])
print("Testing Records:", X_test.shape[0])


# 9. Train Model

model.fit(X_train, y_train)

print("\nModel Training Completed Successfully!")


# 10. Predictions

y_pred = model.predict(X_test)


# 11. Model Evaluation

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\n========== MODEL EVALUATION ==========")

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# 12. Confusion Matrix

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

plt.figure(figsize=(7, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["No Churn", "Churn"],
    yticklabels=["No Churn", "Churn"]
)

plt.title("Customer Churn Prediction - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()

plt.savefig(
    "week3_confusion_matrix.png",
    dpi=300
)

plt.show()


# 13. Feature Importance using Permutation Importance

importance = permutation_importance(
    model,
    X_test,
    y_test,
    scoring="recall",
    n_repeats=5,
    random_state=42,
    n_jobs=-1
)

feature_importance = pd.DataFrame({
    "Feature": X_test.columns,
    "Importance": importance.importances_mean
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print("\n========== TOP 10 IMPORTANT FEATURES ==========")

print(feature_importance.head(10))

feature_importance.to_csv(
    "week3_feature_importance.csv",
    index=False
)

# Feature importance graph
plt.figure(figsize=(10, 6))

sns.barplot(
    data=feature_importance.head(10),
    x="Importance",
    y="Feature"
)

plt.title("Top 10 Features for Customer Churn Prediction")
plt.xlabel("Permutation Importance (Recall Decrease)")
plt.ylabel("Features")
plt.tight_layout()

plt.savefig(
    "week3_feature_importance.png",
    dpi=300
)

plt.show()


# 14. Save Model Evaluation Results

results = pd.DataFrame({
    "Metric": [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score"
    ],
    "Score": [
        accuracy,
        precision,
        recall,
        f1
    ]
})

results.to_csv(
    "week3_model_results.csv",
    index=False
)

print("\nModel evaluation results saved successfully!")

print("\n========== WEEK 3 TASK COMPLETED ==========")
