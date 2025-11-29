import os
import warnings

warnings.filterwarnings("ignore")

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    RocCurveDisplay,
)
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import xgboost as xgb


# LOAD ENGINEERED DATASET
df = pd.read_csv("../data/engineered/featuure_engineering_ready.csv")

X = df.drop(columns=["label"])
y = df["label"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# Create results directory
RESULTS_DIR = "../data/modeling"
os.makedirs(RESULTS_DIR, exist_ok=True)


# EVALUATION FUNCTION
def evaluate_and_save(model_name, y_true, y_pred, y_prob):

    metrics = {
        "Model": model_name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1 Score": f1_score(y_true, y_pred),
        "ROC-AUC": roc_auc_score(y_true, y_prob),
    }

    # Save metrics to CSV
    pd.DataFrame([metrics]).to_csv(
        os.path.join(RESULTS_DIR, f"{model_name}_metrics.csv"), index=False
    )

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, cmap="Blues", fmt="d")
    plt.title(f"{model_name} - Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"{model_name}_confusion_matrix.png"))
    plt.close()

    # ROC Curve
    RocCurveDisplay.from_predictions(y_true, y_prob)
    plt.title(f"{model_name} - ROC Curve")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"{model_name}_roc_curve.png"))
    plt.close()

    print(f"\n===== {model_name} METRICS =====")
    print(pd.DataFrame([metrics]))

    return metrics


# LOGISTIC REGRESSION
log_reg = LogisticRegression(max_iter=5000)
log_reg.fit(X_train, y_train)

y_pred_lr = log_reg.predict(X_test)
y_prob_lr = log_reg.predict_proba(X_test)[:, 1]

evaluate_and_save("Logistic_Regression", y_test, y_pred_lr, y_prob_lr)

joblib.dump(log_reg, os.path.join(RESULTS_DIR, "LogisticRegression_model.joblib"))


# RANDOM FOREST
rf = RandomForestClassifier(
    n_estimators=300, max_depth=25, random_state=42, class_weight="balanced"
)

rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)
y_prob_rf = rf.predict_proba(X_test)[:, 1]

evaluate_and_save("Random_Forest", y_test, y_pred_rf, y_prob_rf)

joblib.dump(rf, os.path.join(RESULTS_DIR, "RandomForest_model.joblib"))


# XGBOOST CLASSIFIER
xgb_model = xgb.XGBClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.9,
    colsample_bytree=0.9,
    eval_metric="logloss",
    random_state=42,
)

xgb_model.fit(X_train, y_train)

y_pred_xgb = xgb_model.predict(X_test)
y_prob_xgb = xgb_model.predict_proba(X_test)[:, 1]

evaluate_and_save("XGBoost", y_test, y_pred_xgb, y_prob_xgb)

joblib.dump(xgb_model, os.path.join(RESULTS_DIR, "XGBoost_model.joblib"))


# STACKING ENSEMBLE MODEL
base_learners = [
    ("lr", LogisticRegression(max_iter=5000)),
    ("rf", RandomForestClassifier(n_estimators=250, max_depth=20, random_state=42)),
    (
        "xgb",
        xgb.XGBClassifier(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=42,
        ),
    ),
]

meta_learner = LogisticRegression(max_iter=5000)

stack_model = StackingClassifier(
    estimators=base_learners,
    final_estimator=meta_learner,
    stack_method="predict_proba",
    passthrough=True,
    n_jobs=-1,
)

print("\nTraining Stacking Ensemble...")
stack_model.fit(X_train, y_train)

y_pred_stack = stack_model.predict(X_test)
y_prob_stack = stack_model.predict_proba(X_test)[:, 1]

evaluate_and_save("Stacking_Ensemble", y_test, y_pred_stack, y_prob_stack)

joblib.dump(stack_model, os.path.join(RESULTS_DIR, "StackingEnsemble_model.joblib"))