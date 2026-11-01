
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "dataset", "customers.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
GRAPH_DIR = os.path.join(OUTPUT_DIR, "graphs")
MODEL_DIR = os.path.join(BASE_DIR, "model")

os.makedirs(GRAPH_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

print("CUSTOMER SPENDING PREDICTION PROJECT")

# STEP 1: Load Dataset

df = pd.read_csv(DATA_PATH)

print("\nFirst 5 Customer Records:")
print(df.head())

print("\nDataset Information:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

# STEP 2: Data Cleaning

df = df.drop_duplicates()

required = [
    "Gender",
    "Age",
    "Annual Income (k$)",
    "Spending Score (1-100)"
]

for col in ["Age", "Annual Income (k$)", "Spending Score (1-100)"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df["Gender"] = df["Gender"].astype("string").str.strip().str.title()

df = df.dropna(subset=required)

print("\nCleaned Dataset:")
print(df.head())

# STEP 3: Create Spending Category

def spending_category(score):
    if score <= 33:
        return "Low"
    elif score <= 66:
        return "Medium"
    else:
        return "High"

df["Spending Category"] = df[
    "Spending Score (1-100)"
].apply(spending_category)

df.to_csv(
    os.path.join(OUTPUT_DIR, "cleaned_customers.csv"),
    index=False
)

print("\nSpending Categories:")
print(df["Spending Category"].value_counts())

# STEP 4: Data Visualization

sns.set_theme()

plt.figure(figsize=(8, 5))
sns.histplot(df["Age"], bins=15, kde=True)
plt.title("Customer Age Distribution")
plt.savefig(os.path.join(GRAPH_DIR, "01_age.png"))
plt.close()

plt.figure(figsize=(8, 5))
sns.countplot(data=df, x="Gender")
plt.title("Gender Distribution")
plt.savefig(os.path.join(GRAPH_DIR, "02_gender.png"))
plt.close()

plt.figure(figsize=(8, 5))
sns.histplot(df["Annual Income (k$)"], kde=True)
plt.title("Annual Income Distribution")
plt.savefig(os.path.join(GRAPH_DIR, "03_income.png"))
plt.close()

plt.figure(figsize=(8, 5))
sns.countplot(
    data=df,
    x="Spending Category",
    order=["Low", "Medium", "High"]
)
plt.title("Spending Category Distribution")
plt.savefig(os.path.join(GRAPH_DIR, "04_category.png"))
plt.close()

plt.figure(figsize=(8, 5))
sns.scatterplot(
    data=df,
    x="Annual Income (k$)",
    y="Spending Score (1-100)",
    hue="Spending Category"
)
plt.title("Income vs Spending Score")
plt.savefig(os.path.join(GRAPH_DIR, "05_income_spending.png"))
plt.close()

print("\nFive graphs created successfully!")

# STEP 5: Feature Engineering and Preprocessing

X = df[["Gender", "Age", "Annual Income (k$)"]]
y = df["Spending Category"]

numeric_features = ["Age", "Annual Income (k$)"]
categorical_features = ["Gender"]

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# STEP 6: Train Machine Learning Models

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(
        max_depth=5,
        random_state=42,
        class_weight="balanced"
    )
}

results = []
trained_models = {}

for name, classifier in models.items():

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", classifier)
    ])

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    trained_models[name] = pipeline

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(
            y_test, predictions,
            average="weighted",
            zero_division=0
        ),
        "Recall": recall_score(
            y_test, predictions,
            average="weighted",
            zero_division=0
        ),
        "F1 Score": f1_score(
            y_test, predictions,
            average="weighted",
            zero_division=0
        )
    })

    print("\nModel:", name)
    print(classification_report(
        y_test, predictions,
        zero_division=0
    ))

# STEP 7: Model Comparison

results_df = pd.DataFrame(results)

results_df.to_csv(
    os.path.join(OUTPUT_DIR, "model_comparison.csv"),
    index=False
)

print("\nMODEL COMPARISON")
print(results_df)

best_name = results_df.sort_values(
    "F1 Score",
    ascending=False
).iloc[0]["Model"]

best_model = trained_models[best_name]

print("\nBest Model:", best_name)

# STEP 8: Confusion Matrix

predictions = best_model.predict(X_test)

cm = confusion_matrix(
    y_test,
    predictions,
    labels=["Low", "Medium", "High"]
)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Low", "Medium", "High"]
)

display.plot(cmap="Blues")
plt.title("Confusion Matrix")
plt.savefig(os.path.join(GRAPH_DIR, "06_confusion_matrix.png"))
plt.close()

# STEP 9: Feature Importance

tree_model = trained_models["Decision Tree"]

feature_names = tree_model.named_steps[
    "preprocessor"
].get_feature_names_out()

importance = tree_model.named_steps[
    "classifier"
].feature_importances_

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importance
}).sort_values("Importance", ascending=False)

importance_df.to_csv(
    os.path.join(OUTPUT_DIR, "feature_importance.csv"),
    index=False
)

plt.figure(figsize=(8, 5))
sns.barplot(
    data=importance_df,
    x="Importance",
    y="Feature"
)
plt.title("Feature Importance")
plt.savefig(os.path.join(GRAPH_DIR, "07_feature_importance.png"))
plt.close()

# STEP 10: Save Best Model

joblib.dump(
    best_model,
    os.path.join(MODEL_DIR, "best_customer_spending_model.pkl")
)

# STEP 11: Sample Prediction

def predict_customer(gender, age, annual_income):

    customer = pd.DataFrame([{
        "Gender": gender,
        "Age": age,
        "Annual Income (k$)": annual_income
    }])

    prediction = best_model.predict(customer)[0]

    probability = best_model.predict_proba(customer)[0].max()

    return prediction, probability * 100

category, confidence = predict_customer(
    "Female", 30, 70
)

print("\nSAMPLE CUSTOMER PREDICTION")
print("Gender: Female")
print("Age: 30")
print("Annual Income: 70")
print("Predicted Category:", category)
print("Confidence: {:.2f}%".format(confidence))

# STEP 12: Business Insight Report

report = f"""
AI-BASED CUSTOMER SPENDING PREDICTION

Total Customers: {len(df)}

Best Machine Learning Model: {best_name}

Most Common Spending Category:
{df['Spending Category'].mode()[0]}

Model Comparison:
{results_df.to_string(index=False)}

Business Recommendations:

1. Give discounts to low spending customers.
2. Recommend suitable products to medium spending customers.
3. Provide loyalty benefits to high spending customers.

This project is created for educational purposes.
"""

with open(
    os.path.join(OUTPUT_DIR, "business_insight_report.txt"),
    "w",
    encoding="utf-8"
) as file:
    file.write(report)

print("\nPROJECT COMPLETED SUCCESSFULLY!")
print("Check the output and model folders.")
