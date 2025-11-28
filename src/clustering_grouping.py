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

