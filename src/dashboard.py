# Enhanced Phishing Detection Dashboard
# Combining modern UI with comprehensive ML pipeline
# Run: streamlit run dashboard.py

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import re
import random
import warnings

warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import (
    KMeans,
    DBSCAN,
    AgglomerativeClustering,
    SpectralClustering,
    Birch,
    OPTICS,
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

# Optional packages check
try:
    from xgboost import XGBClassifier

    _HAS_XGBOOST = True
except:
    _HAS_XGBOOST = False

try:
    import tensorflow as tf
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Embedding, SpatialDropout1D, LSTM, Dense

    _HAS_TF = True
except:
    _HAS_TF = False

try:
    from transformers import pipeline

    _HAS_TRANSFORMERS = True
except:
    _HAS_TRANSFORMERS = False

# Set seeds
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
if _HAS_TF:
    try:
        tf.random.set_seed(RANDOM_SEED)
    except:
        pass


# ===== UTILITY FUNCTIONS =====
def clean_text(text: str) -> str:
    """Anonymize and normalize text"""
    text = str(text).lower()
    text = re.sub(r"http\S+", " link ", text)
    text = re.sub(r"\S+@\S+", " email ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def engineer_features(df):
    """Create engineered features"""
    df = df.copy()
    df["Email_Content"] = df.get("Email_Content", "").fillna("").astype(str)
    df["Email_Subject"] = df.get("Email_Subject", "").fillna("").astype(str)
    df["clean_content"] = df["Email_Content"].apply(clean_text)
    df["subject_length"] = df["Email_Subject"].apply(lambda x: len(str(x)))
    df["link_count"] = df["Email_Content"].apply(
        lambda x: len(re.findall(r"http\S+", str(x)))
    )
    df["has_link"] = (df["link_count"] > 0).astype(int)
    df["word_count"] = df["clean_content"].apply(lambda x: len(x.split()))
    return df


def add_laplace_dp(df, numeric_columns, epsilon=1.0):
    """Add Laplace noise for DP"""
    df = df.copy()
    scale = 1.0 / max(epsilon, 1e-9)
    for c in numeric_columns:
        noise = np.random.laplace(0, scale, df.shape[0])
        df[c] = df[c] + noise
    return df


def compute_metrics(y_true, y_pred, y_proba=None):
    """Compute classification metrics"""
    try:
        y_true = np.array(y_true).astype(int)
        y_pred = np.array(y_pred).astype(int)
    except:
        le = LabelEncoder()
        y_true = le.fit_transform(y_true)
        y_pred = le.transform(y_pred)

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": (
            float(roc_auc_score(y_true, y_proba)) if y_proba is not None else None
        ),
    }


# ===== PAGE CONFIG =====
st.set_page_config(page_title="CyberSecure Analytics", page_icon="🛡️", layout="wide")

# ===== MODERN CSS =====
st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    * { font-family: 'Inter', sans-serif; }
    .main { background: linear-gradient(135deg, #0a0a0a 0%, #1a1a2e 100%); color: #e5e5e5; }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0f23 0%, #1a1a2e 100%);
        border-right: 1px solid rgba(99, 102, 241, 0.2);
    }
    .stButton>button {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #ec4899 100%);
        color: white; border: none; border-radius: 0.75rem; padding: 0.75rem 1.5rem;
        font-weight: 600; box-shadow: 0 4px 20px rgba(99, 102, 241, 0.4);
    }
    .stButton>button:hover { transform: translateY(-2px); }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        color: white; box-shadow: 0 4px 15px rgba(99, 102, 241, 0.4);
    }
</style>
""",
    unsafe_allow_html=True,
)

# ===== SESSION STATE =====
if "df" not in st.session_state:
    st.session_state.df = None
if "models" not in st.session_state:
    st.session_state.models = {}
if "results" not in st.session_state:
    st.session_state.results = {}

# ===== HEADER =====
st.markdown(
    """
<div style='display: flex; align-items: center; gap: 1.5rem; margin-bottom: 2rem; padding: 1.5rem; 
            background: linear-gradient(135deg, rgba(15, 23, 42, 0.9), rgba(31, 41, 55, 0.9)); 
            border-radius: 1.5rem; border: 1px solid rgba(99, 102, 241, 0.3);'>
    <div style='background: linear-gradient(135deg, #6366f1, #ec4899); padding: 1rem; border-radius: 1rem;'>
        <span style='font-size: 2rem;'>🛡️</span>
    </div>
    <div style='flex: 1;'>
        <h1 style='margin: 0; background: linear-gradient(135deg, #6366f1, #ec4899); 
                   -webkit-background-clip: text; -webkit-text-fill-color: transparent; 
                   font-size: 2rem; font-weight: 800;'>CyberSecure Analytics</h1>
        <p style='margin: 0.5rem 0 0; color: #9ca3af; font-weight: 500;'>
            Advanced AI Phishing Detection • Full ML Pipeline • Privacy-Preserving</p>
    </div>
</div>
""",
    unsafe_allow_html=True,
)

# ===== SIDEBAR =====
with st.sidebar:
    st.markdown("### 📤 Data Input")
    uploaded = st.file_uploader(
        "Upload CSV", type=["csv"], label_visibility="collapsed"
    )
    use_local = st.checkbox("Use local sample file")

    if uploaded:
        st.session_state.df = pd.read_csv(uploaded)
        st.success(f"✓ {uploaded.name}")
    elif use_local and os.path.exists("Cleaned_PhishingEmailData.csv"):
        st.session_state.df = pd.read_csv("Cleaned_PhishingEmailData.csv")
        st.success("✓ Local file loaded")

    st.markdown("---")
    st.markdown("### 🔒 Privacy Options")
    privacy = st.selectbox("Privacy", ["None", "Laplace DP", "Federated Learning"])
    if privacy == "Laplace DP":
        epsilon = st.slider("ε (epsilon)", 0.1, 5.0, 1.0)

    st.markdown("---")
    st.markdown("**Required Columns:**")
    st.code("• Email_Content\n• Email_Subject\n• Label")

    st.markdown(
        f"""
    <div style='margin-top: 2rem; padding: 1rem; background: rgba(15, 23, 42, 0.5); border-radius: 0.75rem;'>
        <div style='display: flex; justify-content: space-between;'>
            <span style='color: #9ca3af;'>Status:</span>
            <span style='color: {'#22c55e' if st.session_state.df is not None else '#9ca3af'}; font-weight: 700;'>
                {'READY' if st.session_state.df is not None else 'NO DATA'}
            </span>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

# ===== MAIN CONTENT =====
if st.session_state.df is None:
    st.info("📤 Please upload a dataset in the sidebar")
    st.stop()

df = st.session_state.df

# Validate
required = ["Email_Content", "Email_Subject", "Label"]
missing = [c for c in required if c not in df.columns]
if missing:
    st.error(f"Missing columns: {missing}")
    st.stop()

# Engineer features
df = engineer_features(df)
feature_cols = ["subject_length", "link_count", "has_link", "word_count"]

# Apply privacy
if privacy == "Laplace DP":
    df[feature_cols] = add_laplace_dp(df[feature_cols], feature_cols, epsilon)

# Checks
label_counts = df["Label"].value_counts()
can_stratify = label_counts.min() >= 2
allow_heavy = len(df) >= 50 and label_counts.min() >= 5

# ===== TABS =====
tabs = st.tabs(
    ["📊 Overview", "🔍 EDA", "🤖 Training", "📈 Evaluation", "🧪 Live Test"]
)

# TAB 1: Overview
with tabs[0]:
    col1, col2, col3, col4 = st.columns(4)

    metrics_data = [
        ("📈", "Total Emails", f"{len(df):,}", "#6366f1"),
        ("⚠️", "Phishing", f"{(df['Label']==1).sum():,}", "#ef4444"),
        ("✓", "Legitimate", f"{(df['Label']==0).sum():,}", "#22c55e"),
        ("🎯", "Features", f"{len(feature_cols)}", "#f59e0b"),
    ]

    for col, (icon, title, value, color) in zip([col1, col2, col3, col4], metrics_data):
        with col:
            st.markdown(
                f"""
            <div style='padding: 1.5rem; border-radius: 1rem; background: rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, 0.1);
                        border: 1px solid rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, 0.3);'>
                <div style='font-size: 1.5rem; margin-bottom: 0.5rem;'>{icon}</div>
                <div style='color: #9ca3af; font-size: 0.875rem; text-transform: uppercase; margin-bottom: 0.5rem;'>{title}</div>
                <div style='font-size: 2rem; font-weight: 800; color: {color};'>{value}</div>
            </div>
            """,
                unsafe_allow_html=True,
            )

    st.markdown("### 📄 Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        fig = px.pie(
            values=label_counts.values,
            names=label_counts.index,
            title="Label Distribution",
            color_discrete_sequence=["#22c55e", "#ef4444"],
        )
        fig.update_layout(
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", height=300
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("#### Feature Statistics")
        st.dataframe(df[feature_cols].describe(), use_container_width=True)

# TAB 2: EDA
with tabs[1]:
    st.markdown("## 🔍 Exploratory Data Analysis")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Feature Distributions")
        for feat in feature_cols:
            fig = px.histogram(
                df, x=feat, title=feat, color_discrete_sequence=["#6366f1"]
            )
            fig.update_layout(
                template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", height=250
            )
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### Correlation Heatmap")
        corr_data = df[feature_cols + ["Label"]].copy()
        corr_data["Label"] = LabelEncoder().fit_transform(corr_data["Label"])
        fig = px.imshow(
            corr_data.corr(), text_auto=True, color_continuous_scale="RdBu_r"
        )
        fig.update_layout(
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", height=400
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 🎯 Clustering Visualization")
    cluster_algo = st.selectbox(
        "Algorithm", ["KMeans", "DBSCAN", "Agglomerative", "Spectral"]
    )
    n_clusters = st.slider("Clusters", 2, 10, 4) if cluster_algo != "DBSCAN" else 0

    X = df[feature_cols].fillna(0).values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    if cluster_algo == "KMeans":
        clusterer = KMeans(n_clusters=n_clusters, random_state=RANDOM_SEED)
    elif cluster_algo == "DBSCAN":
        eps = st.slider("eps", 0.1, 5.0, 0.5)
        clusterer = DBSCAN(eps=eps, min_samples=5)
    elif cluster_algo == "Agglomerative":
        clusterer = AgglomerativeClustering(n_clusters=n_clusters)
    else:
        clusterer = SpectralClustering(n_clusters=n_clusters, random_state=RANDOM_SEED)

    clusters = clusterer.fit_predict(X_scaled)

    pca = PCA(n_components=2, random_state=RANDOM_SEED)
    X_2d = pca.fit_transform(X_scaled)

    cluster_df = pd.DataFrame(
        {
            "PC1": X_2d[:, 0],
            "PC2": X_2d[:, 1],
            "Cluster": clusters.astype(str),
            "Label": df["Label"].astype(str),
        }
    )

    fig = px.scatter(
        cluster_df,
        x="PC1",
        y="PC2",
        color="Cluster",
        symbol="Label",
        title=f"{cluster_algo} Clustering",
    )
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", height=500)
    st.plotly_chart(fig, use_container_width=True)

# TAB 3: Training
with tabs[2]:
    st.markdown("## 🤖 Model Training")

    available = ["LogisticRegression", "RandomForest", "NaiveBayes"]
    if _HAS_XGBOOST and allow_heavy:
        available.append("XGBoost")
    if _HAS_TF and allow_heavy:
        available.append("LSTM")

    selected = st.multiselect(
        "Select models", available, default=["LogisticRegression", "RandomForest"]
    )

    test_size = st.slider("Test size (%)", 10, 40, 20) / 100

    if st.button("🚀 Train Selected Models"):
        with st.spinner("Training models..."):
            X = df[feature_cols].fillna(0)
            le = LabelEncoder()
            y = le.fit_transform(df["Label"])

            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X,
                    y,
                    test_size=test_size,
                    random_state=RANDOM_SEED,
                    stratify=y if can_stratify else None,
                )
            except:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=RANDOM_SEED
                )

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            st.session_state.models = {}
            st.session_state.results = {}

            # Train models
            if "LogisticRegression" in selected:
                lr = LogisticRegression(max_iter=1000)
                lr.fit(X_train_scaled, y_train)
                y_pred = lr.predict(X_test_scaled)
                y_proba = lr.predict_proba(X_test_scaled)[:, 1]
                st.session_state.models["LogisticRegression"] = {
                    "model": lr,
                    "scaler": scaler,
                    "le": le,
                }
                st.session_state.results["LogisticRegression"] = compute_metrics(
                    y_test, y_pred, y_proba
                )

            if "RandomForest" in selected:
                rf = RandomForestClassifier(n_estimators=200, random_state=RANDOM_SEED)
                rf.fit(X_train_scaled, y_train)
                y_pred = rf.predict(X_test_scaled)
                y_proba = rf.predict_proba(X_test_scaled)[:, 1]
                st.session_state.models["RandomForest"] = {
                    "model": rf,
                    "scaler": scaler,
                    "le": le,
                }
                st.session_state.results["RandomForest"] = compute_metrics(
                    y_test, y_pred, y_proba
                )

            if "NaiveBayes" in selected:
                nb = GaussianNB()
                nb.fit(X_train_scaled, y_train)
                y_pred = nb.predict(X_test_scaled)
                y_proba = nb.predict_proba(X_test_scaled)[:, 1]
                st.session_state.models["NaiveBayes"] = {
                    "model": nb,
                    "scaler": scaler,
                    "le": le,
                }
                st.session_state.results["NaiveBayes"] = compute_metrics(
                    y_test, y_pred, y_proba
                )

            if "XGBoost" in selected and _HAS_XGBOOST:
                xgb = XGBClassifier(
                    use_label_encoder=False,
                    eval_metric="logloss",
                    random_state=RANDOM_SEED,
                )
                xgb.fit(X_train_scaled, y_train)
                y_pred = xgb.predict(X_test_scaled)
                y_proba = xgb.predict_proba(X_test_scaled)[:, 1]
                st.session_state.models["XGBoost"] = {
                    "model": xgb,
                    "scaler": scaler,
                    "le": le,
                }
                st.session_state.results["XGBoost"] = compute_metrics(
                    y_test, y_pred, y_proba
                )

            st.success("✅ Training complete!")
            st.rerun()

# TAB 4: Evaluation
with tabs[3]:
    st.markdown("## 📈 Model Evaluation")

    if st.session_state.results:
        results_df = pd.DataFrame(st.session_state.results).T
        results_df = results_df[["accuracy", "precision", "recall", "f1", "roc_auc"]]

        st.dataframe(
            results_df.style.highlight_max(axis=0, color="lightgreen"),
            use_container_width=True,
        )

        # Metrics comparison
        for metric in ["accuracy", "precision", "recall", "f1"]:
            fig = px.bar(
                results_df,
                y=metric,
                title=f"{metric.capitalize()} Comparison",
                color_discrete_sequence=["#6366f1"],
            )
            fig.update_layout(
                template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", height=300
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Train models first in the Training tab")

# TAB 5: Live Testing
with tabs[4]:
    st.markdown("## 🧪 Live Email Testing")

    email_text = st.text_area(
        "Paste email content",
        height=200,
        placeholder="Subject: Urgent Account Verification\n\nDear user, click here immediately...",
    )

    model_choice = st.selectbox(
        "Select model", ["None"] + list(st.session_state.models.keys())
    )

    if st.button("🔍 Analyze Email"):
        if not email_text.strip():
            st.warning("Please enter email content")
        elif model_choice == "None":
            st.warning("Please select a trained model")
        else:
            # Extract features
            subj_len = (
                len(email_text.split("\n")[0])
                if "\n" in email_text
                else len(email_text)
            )
            link_count = len(re.findall(r"http\S+", email_text))
            has_link = 1 if link_count > 0 else 0
            word_count = len(clean_text(email_text).split())

            feat = np.array([[subj_len, link_count, has_link, word_count]])

            model_data = st.session_state.models[model_choice]
            feat_scaled = model_data["scaler"].transform(feat)

            pred = model_data["model"].predict(feat_scaled)[0]
            proba = model_data["model"].predict_proba(feat_scaled)[0, 1]
            label = model_data["le"].inverse_transform([int(pred)])[0]

            is_phishing = label == 1

            if is_phishing:
                st.markdown(
                    f"""
                <div style='background: rgba(239, 68, 68, 0.1); padding: 2rem; border-radius: 1rem; 
                            border: 1px solid rgba(239, 68, 68, 0.3);'>
                    <h2 style='color: #ef4444;'>⚠️ PHISHING DETECTED</h2>
                    <p style='font-size: 1.25rem;'>Confidence: {proba*100:.1f}%</p>
                    <div style='margin-top: 1rem; display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem;'>
                        <div style='text-align: center;'>
                            <div style='font-size: 1.5rem;'>{'✗' if has_link else '✓'}</div>
                            <div style='color: #9ca3af;'>Links</div>
                        </div>
                        <div style='text-align: center;'>
                            <div style='font-size: 1.5rem;'>{link_count}</div>
                            <div style='color: #9ca3af;'>Link Count</div>
                        </div>
                        <div style='text-align: center;'>
                            <div style='font-size: 1.5rem;'>{word_count}</div>
                            <div style='color: #9ca3af;'>Words</div>
                        </div>
                    </div>
                </div>
                """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""
                <div style='background: rgba(34, 197, 94, 0.1); padding: 2rem; border-radius: 1rem; 
                            border: 1px solid rgba(34, 197, 94, 0.3);'>
                    <h2 style='color: #22c55e;'>✓ LEGITIMATE EMAIL</h2>
                    <p style='font-size: 1.25rem;'>Confidence: {(1-proba)*100:.1f}%</p>
                    <div style='margin-top: 1rem; display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem;'>
                        <div style='text-align: center;'>
                            <div style='font-size: 1.5rem;'>{'✗' if has_link else '✓'}</div>
                            <div style='color: #9ca3af;'>Links</div>
                        </div>
                        <div style='text-align: center;'>
                            <div style='font-size: 1.5rem;'>{link_count}</div>
                            <div style='color: #9ca3af;'>Link Count</div>
                        </div>
                        <div style='text-align: center;'>
                            <div style='font-size: 1.5rem;'>{word_count}</div>
                            <div style='color: #9ca3af;'>Words</div>
                        </div>
                    </div>
                </div>
                """,
                    unsafe_allow_html=True,
                )
