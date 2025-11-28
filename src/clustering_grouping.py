import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import os


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


# HIERARCHICAL CLUSTERING (Agglomerative)
hier = AgglomerativeClustering(n_clusters=3)
hier_labels = hier.fit_predict(X_np)

try:
    print("Silhouette (Hierarchical):", silhouette_score(X_np, hier_labels))
except:
    print("Silhouette (Hierarchical): Not valid")


# DIMENSIONALITY REDUCTION FOR VISUALISATION
pca_2d = do_pca(X_np)
tsne_2d = do_tsne(X_np)


# PCA Visualizations
plot_clusters(pca_2d, kmeans_labels, "K-Means Clusters (PCA)")
plot_clusters(pca_2d, dbscan_labels, "DBSCAN Clusters (PCA)")
plot_clusters(pca_2d, hier_labels, "Hierarchical Clusters (PCA)")

# t-SNE Visualizations
plot_clusters(tsne_2d, kmeans_labels, "K-Means Clusters (t-SNE)")
plot_clusters(tsne_2d, dbscan_labels, "DBSCAN Clusters (t-SNE)")
plot_clusters(tsne_2d, hier_labels, "Hierarchical Clusters (t-SNE)")


# SAVE CLEAN CLUSTERED DATASET
df_clean = df.copy()
df_clean["Cluster_KMeans"] = kmeans_labels
df_clean["Cluster_DBSCAN"] = dbscan_labels
df_clean["Cluster_Hierarchical"] = hier_labels

# Confirm no missing values
print(df_clean.isna().sum())

# Save final dataset
df_clean.to_csv("../data/clustered/Nazario_clusteredData.csv", index=False)

print(df_clean.info())


# K-MEANS CLUSTER ANALYSIS
# Use only metadata features for analysis
metadata_cols = [c for c in X.columns if not c.startswith("svd_")]
cluster_summary = df_clean.copy()
cluster_summary["cluster"] = kmeans_labels

# Compute mean values per cluster
cluster_means = cluster_summary.groupby("cluster")[metadata_cols].mean()
cluster_means.to_csv(
    os.path.join("../data/clustered", "kmeans_cluster_characteristics.csv")
)

# Top distinguishing features per cluster
top_features = {}
for cluster_id in cluster_means.index:
    diffs = (cluster_means.loc[cluster_id] - cluster_means.mean()).abs()
    top = diffs.sort_values(ascending=False).head(10)
    top_features[cluster_id] = top.index.tolist()

# Save top features
with open(
    os.path.join("../data/clustered", "kmeans_cluster_top_features.txt"), "w"
) as f:
    for cluster_id, feats in top_features.items():
        f.write(f"\nCluster {cluster_id} top features:\n")
        for feat in feats:
            f.write(f" - {feat}\n")


# Human-readable interpretation
def interpret_cluster(row):
    desc = []
    if row["url_count"] > cluster_means["url_count"].mean():
        desc.append("contains many URLs")
    if row["body_length"] < cluster_means["body_length"].mean():
        desc.append("short email body")
    if row["subject_length"] > cluster_means["subject_length"].mean():
        desc.append("long subject line")
    if row["sender_phish_score"] < 0.4:
        desc.append("sent from low-trust sender domain")
    if row["total_suspicious_keywords"] > 2:
        desc.append("contains many phishing-related keywords")
    return ", ".join(desc) if desc else "no strong characteristics"


interpretations = {}
for cid in cluster_means.index:
    interpretations[cid] = interpret_cluster(cluster_means.loc[cid])

# Save interpretations
with open(
    os.path.join("../data/clustered", "kmeans_cluster_interpretations.txt"), "w"
) as f:
    for cid, text in interpretations.items():
        f.write(f"Cluster {cid}: {text}\n")
