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
RESULTS_DIR = "..data/modeling/models/"
os.makedirs(RESULTS_DIR, exist_ok=True)

