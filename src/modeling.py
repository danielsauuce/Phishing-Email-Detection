import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report,
)

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier


# Load Engineered Dataset
df = pd.read_csv("../data/engineered/featuure_engineering_ready.csv")
print(df.info())

X = df.drop("label", axis=1)
y = df["label"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)


# Define Models

models = {
    "RandomForest": RandomForestClassifier(
        n_estimators=300, max_depth=20, random_state=42
    ),
    "LogisticRegression": LogisticRegression(max_iter=2000, solver="lbfgs"),
    "XGBoost": XGBClassifier(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=8,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="logloss",
        random_state=42,
    ),
}

results = []


# Created a results directory
os.makedirs("../results", exist_ok=True)


# Train, Predict & Evaluate

for name, model in models.items():
    print(f"\n==============================")
    print(f" TRAINING MODEL: {name} ")
    print(f"==============================")

    # Train
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    # Store results
    results.append([name, acc, prec, rec, f1, auc])

    # Save Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"{name} - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(f"../results/{name}_confusion_matrix.png")
    plt.close()

    # Save ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.title(f"{name} - ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.tight_layout()
    plt.savefig(f"../results/{name}_roc_curve.png")
    plt.close()

    # Save Classification Report
    with open(f"../results/{name}_classification_report.txt", "w") as f:
        f.write(classification_report(y_test, y_pred))


# Save Performance Table
df_results = pd.DataFrame(
    results, columns=["Model", "Accuracy", "Precision", "Recall", "F1", "ROC_AUC"]
)

df_results.to_csv("../results/model_performance.csv", index=False)
print("\nModel performance saved to ../results/model_performance.csv")

print("\nAll models trained and evaluated successfully.")
