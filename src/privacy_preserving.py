import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load engineered dataset
df = pd.read_csv("../data/engineered/featuure_engineering_ready.csv")

X = df.drop(columns=["label"])
y = df["label"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


def add_differential_privacy_noise(X, epsilon=1.0):

    # Simulates differential privacy by adding Gaussian noise to feature values.
    sensitivity = X.max() - X.min()
    noise = np.random.normal(loc=0, scale=sensitivity / epsilon, size=X.shape)
    X_noisy = X + noise
    return X_noisy


