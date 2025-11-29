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


def simulate_federated_learning(X_train, y_train, n_clients=3):
    """
    Simulates federated learning:
    - Split data across 'clients'
    - Train separate models locally
    - Aggregate predictions without sharing raw data
    """
    client_models = []
    split_size = len(X_train) // n_clients
    for i in range(n_clients):
        start = i * split_size
        end = (i + 1) * split_size if i < n_clients - 1 else len(X_train)
        X_client = X_train.iloc[start:end]
        y_client = y_train.iloc[start:end]

        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X_client, y_client)
        client_models.append(model)
        print(f"- Trained client {i+1} model on {len(X_client)} samples")

    return client_models


