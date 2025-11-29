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


def aggregate_predictions(client_models, X_test):
    """
    Aggregates predictions from multiple client models (simple averaging).
    """
    probs = np.zeros((len(X_test), 2))
    for model in client_models:
        probs += model.predict_proba(X_test)
    probs /= len(client_models)
    y_pred = np.argmax(probs, axis=1)
    return y_pred, probs[:, 1]


def simulate_privacy_training():
    print("Simulating privacy-preserving training...")

    # Step 1: Apply differential privacy to training features
    X_train_noisy = add_differential_privacy_noise(X_train, epsilon=1.0)
    print("- Applied differential privacy noise to training data")

    # Step 2: Simulate federated learning
    client_models = simulate_federated_learning(X_train_noisy, y_train, n_clients=3)
    print("- Federated learning simulation complete across 3 clients")

    # Step 3: Aggregate predictions securely
    y_pred, y_prob = aggregate_predictions(client_models, X_test)
    print("- Aggregated client predictions without sharing raw data")

    # Step 4: Evaluate model
    acc = accuracy_score(y_test, y_pred)
    print(f"- Accuracy of aggregated privacy-preserving model: {acc:.4f}")
    print(
        "Privacy-preserving simulation complete. Ready for integration with PySyft or SmartNoise."
    )


def add_differential_privacy_noise(X, epsilon=1.0):
    """
    Adds Gaussian noise to features to simulate differential privacy.
    """
    sensitivity = X.max() - X.min()
    noise = np.random.normal(loc=0, scale=sensitivity / epsilon, size=X.shape)
    X_noisy = X + noise
    return X_noisy


if __name__ == "__main__":
    simulate_privacy_training()
