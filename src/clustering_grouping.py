import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt


# Load engineered dataset
df = pd.read_csv("../data/engineered/featuure_engineering_ready.csv")

# Separate features from labels
X = df.drop(columns=["label"])
y = df["label"]

# Fix problematic values
X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
X_np = X.values


# PCA projection
def do_pca(X, components=2):
    pca = PCA(n_components=components)
    return pca.fit_transform(X)


# t-SNE projection
def do_tsne(X, components=2):
    tsne = TSNE(n_components=components, perplexity=30, learning_rate=200)
    return tsne.fit_transform(X)


# clustering plots
def plot_clusters(embedding, labels, title):
    plt.figure(figsize=(7, 5))
    scatter = plt.scatter(
        embedding[:, 0], embedding[:, 1], c=labels, cmap="viridis", alpha=0.7
    )
    plt.colorbar(scatter)
    plt.title(title)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.tight_layout()
    plt.show()


# K-MEANS CLUSTERING
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans_labels = kmeans.fit_predict(X_np)

try:
    print("Silhouette (KMeans):", silhouette_score(X_np, kmeans_labels))
except:
    print("Silhouette (KMeans): Not valid")


# DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=5)
dbscan_labels = dbscan.fit_predict(X_np)

try:
    print("Silhouette (DBSCAN):", silhouette_score(X_np, dbscan_labels))
except:
    print("Silhouette (DBSCAN): Not valid (clusters may include noise)")
