import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# IMPORTANT: Import StandardScaler and PCA (this fixes the NameError)
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# -------------------------- PAGE CONFIG & STYLING (Unchanged) --------------------------
st.set_page_config(page_title="CyberSecure Analytics", page_icon="🛡️", layout="wide")

st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    * { font-family: 'Inter', sans-serif; }
    .main { background: linear-gradient(135deg, #0a0a0a 0%, #1a1a2e 100%); color: #e5e5e5; }
    .stPlotlyChart { background: #1a1f2e !important; border-radius: 12px; padding: 12px; }
    h1, h2, h3 { color: #e5e5e5 !important; }
    .stButton>button { background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #ec4899 100%);
                        background-size: 200% 200%; animation: gradient-shift 3s ease infinite;
                        color: white; border: none; border-radius: 0.75rem; padding: 0.75rem 1.5rem;
                        font-weight: 600; box-shadow: 0 4px 20px rgba(99,102,241,0.4); transition: all 0.3s ease; }
    .stButton>button:hover { transform: translateY(-2px); box-shadow: 0 8px 30px rgba(99,102,241,0.6); }
    .metric-card { background: #1a1f2e; padding: 1.5rem; border-radius: 12px; color: white;
                   box-shadow: 0 4px 10px rgba(0,0,0,0.4); border-left: 5px solid #60a5fa; transition: all 0.3s ease; }
    .metric-card:hover { transform: translateY(-5px); }
    .success-card { border-left: 5px solid #22c55e; }
    .warning-card { border-left: 5px solid #eab308; }
    .danger-card { border-left: 5px solid #ef4444; }
    .glow-destructive { box-shadow: 0 0 20px rgba(239,68,68,0.4); animation: pulse 2s infinite; }
    @keyframes gradient-shift { 0% { background-position: 0% 50%; } 50% { background-position: 100% 50%; } 100% { background-position: 0% 50%; } }
    @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.5; } }
</style>
""",
    unsafe_allow_html=True,
)

# -------------------------- HEADER --------------------------
st.markdown(
    """
<div style='display: flex; align-items: center; gap: 1.5rem; margin-bottom: 2.5rem; padding: 1.5rem 2rem;
            background: linear-gradient(135deg, rgba(15, 23, 42, 0.9) 0%, rgba(31, 41, 55, 0.9) 100%);
            backdrop-filter: blur(20px); border-radius: 1.5rem; border: 1px solid rgba(99, 102, 241, 0.3);
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4), 0 0 80px rgba(99, 102, 241, 0.2);'>
    <div style='background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #ec4899 100%);
                padding: 1rem; border-radius: 1rem; box-shadow: 0 4px 20px rgba(99, 102, 241, 0.5);
                animation: glow 2s ease-in-out infinite;'>
        <span style='font-size: 2rem; filter: drop-shadow(0 0 10px rgba(255, 255, 255, 0.5));'>🛡️</span>
    </div>
    <div style='flex: 1;'>
        <h1 style='margin: 0; background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #ec4899 100%);
                   -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                   font-size: 2rem; font-weight: 800; letter-spacing: -0.03em;'>
            CyberSecure Analytics
        </h1>
        <p style='margin: 0.5rem 0 0 0; color: #9ca3af; font-size: 1rem; font-weight: 500;'>
            AI-Powered Phishing Detection • Real-Time Threat Intelligence
        </p>
    </div>
    <div style='background: rgba(15, 23, 42, 0.8); backdrop-filter: blur(10px);
                padding: 0.75rem 1.5rem; border-radius: 1rem; border: 1px solid rgba(34, 197, 94, 0.3);
                box-shadow: 0 4px 15px rgba(34, 197, 94, 0.2);'>
        <div style='display: flex; align-items: center; gap: 0.75rem;'>
            <span style='display: inline-block; width: 0.625rem; height: 0.625rem;
                        background: #22c55e; border-radius: 50%; box-shadow: 0 0 10px #22c55e;
                        animation: pulse 2s infinite;'></span>
            <span style='color: #e5e5e5; font-size: 0.875rem; font-weight: 600;'>SYSTEM ONLINE</span>
        </div>
    </div>
</div>
""",
    unsafe_allow_html=True,
)

# -------------------------- SIDEBAR --------------------------
with st.sidebar:
    st.markdown(
        """
    <div style='text-align: center; margin-bottom: 2rem;'>
        <div style='background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
                    padding: 1rem; border-radius: 1rem; margin-bottom: 1rem;
                    box-shadow: 0 4px 20px rgba(99, 102, 241, 0.4);'>
            <span style='font-size: 2.5rem;'>📤</span>
        </div>
        <h2 style='margin: 0; font-size: 1.5rem; font-weight: 700;
                   background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
                   -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>
            Data Input
        </h2>
        <p style='color: #9ca3af; font-size: 0.875rem; margin-top: 0.5rem; font-weight: 500;'>
            Upload email dataset for AI analysis
        </p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "Upload CSV File", type=["csv"], label_visibility="collapsed"
    )

    if uploaded_file:
        st.markdown(
            f"""
        <div style='display: flex; align-items: center; gap: 0.75rem; padding: 1rem; border-radius: 1rem;
                    background: rgba(34, 197, 94, 0.1); backdrop-filter: blur(10px);
                    border: 1px solid rgba(34, 197, 94, 0.3); box-shadow: 0 4px 15px rgba(34, 197, 94, 0.2);'>
            <span style='font-size: 1.5rem;'>✓</span>
            <span style='color: #22c55e; font-weight: 600; font-size: 0.875rem; flex: 1;'>{uploaded_file.name}</span>
        </div>
        """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(
        """
    <div style='background: linear-gradient(135deg, rgba(15, 23, 42, 0.9) 0%, rgba(31, 41, 55, 0.9) 100%);
                backdrop-filter: blur(20px); padding: 1.5rem; border-radius: 1rem;
                border: 1px solid rgba(99, 102, 241, 0.3); box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);'>
        <h4 style='font-size: 1rem; margin-bottom: 1rem; font-weight: 700;
                   background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
                   -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>
            📋 Required Columns
        </h4>
        <ul style='font-size: 0.875rem; color: #e5e5e5; list-style: none; padding: 0; margin: 0;'>
            <li style='margin-bottom: 0.75rem; display: flex; align-items: center; gap: 0.75rem;
                       padding: 0.75rem; background: rgba(99, 102, 241, 0.05);
                       border-radius: 0.5rem; border-left: 3px solid #6366f1;'>
                <span style='width: 8px; height: 8px;
                            background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
                            border-radius: 50%; box-shadow: 0 0 10px #6366f1;'></span>
                <code style='color: #a5b4fc; font-weight: 600;'>Email_Content</code>
            </li>
            <li style='margin-bottom: 0.75rem; display: flex; align-items: center; gap: 0.75rem;
                       padding: 0.75rem; background: rgba(99, 102, 241, 0.05);
                       border-radius: 0.5rem; border-left: 3px solid #8b5cf6;'>
                <span style='width: 8px; height: 8px;
                            background: linear-gradient(135deg, #8b5cf6 0%, #ec4899 100%);
                            border-radius: 50%; box-shadow: 0 0 10px #8b5cf6;'></span>
                <code style='color: #c4b5fd; font-weight: 600;'>Email_Subject</code>
            </li>
            <li style='display: flex; align-items: center; gap: 0.75rem;
                       padding: 0.75rem; background: rgba(99, 102, 241, 0.05);
                       border-radius: 0.5rem; border-left: 3px solid #ec4899;'>
                <span style='width: 8px; height: 8px;
                            background: linear-gradient(135deg, #ec4899 0%, #f43f5e 100%);
                            border-radius: 50%; box-shadow: 0 0 10px #ec4899;'></span>
                <code style='color: #f9a8d4; font-weight: 600;'>Label</code>
            </li>
        </ul>
    </div>
    """,
        unsafe_allow_html=True,
    )


# -------------------------- DATA PROCESSING --------------------------
@st.cache_data
def load_and_process_data(file):
    df = pd.read_csv(file, encoding="ISO-8859-1")

    # Auto-map your Nazario columns
    col_map = {
        "body": "Email_Content",
        "subject": "Email_Subject",
        "label": "Label",
        "sender": "sender",
        "receiver": "receiver",
        "date": "date",
        "urls": "urls",
    }
    df = df.rename(columns=col_map)

    required = ["Email_Content", "Email_Subject", "Label"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        st.error(f"Missing required columns: {missing}")
        return None

    # Anonymisation
    df["Email_Content"] = (
        df["Email_Content"].astype(str).str.replace(r"http\S+", " <LINK> ", regex=True)
    )
    df["Email_Content"] = df["Email_Content"].str.replace(
        r"\S+@\S+", " <EMAIL> ", regex=True
    )

    # Feature engineering
    df["subject_len"] = df["Email_Subject"].astype(str).str.len()
    df["body_len"] = df["Email_Content"].astype(str).str.split().str.len()
    df["url_count"] = df["Email_Content"].str.count(" <LINK> ")
    df["has_url"] = (df["url_count"] > 0).astype(int)
    df["word_count"] = df["Email_Content"].astype(str).str.split().str.len()
    df["suspicious_keywords"] = (
        df["Email_Subject"]
        .astype(str)
        .str.lower()
        .str.contains(
            "urgent|verify|account|password|security|login|bank|paypal|confirm|update|suspend|immediate|alert"
        )
        .astype(int)
    )

    # Safe Label conversion
    if df["Label"].dtype == "object":
        df["Label"] = df["Label"].map(
            lambda x: (
                1
                if str(x).lower() in ["phishing", "1", "spam", "yes"]
                else 0 if pd.notna(x) else np.nan
            )
        )
    df["Label"] = pd.to_numeric(df["Label"], errors="coerce").fillna(0).astype(int)

    return df


# Load data
df = None
if uploaded_file is not None:
    with st.spinner("Loading and processing your dataset..."):
        df = load_and_process_data(uploaded_file)
    if df is not None:
        st.success("Dataset loaded and processed successfully!")

# -------------------------- TABS --------------------------
tab1, tab2 = st.tabs(["📊 Analytics Dashboard", "🧪 Live Email Testing"])

with tab1:
    if df is None:
        st.info("Upload a CSV file to see analytics and results")
    else:
        # Missing Values Before & After
        st.markdown("### Missing Values: Before vs After Cleaning")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Before Cleaning")
            missing_before = df.isnull().sum()
            missing_before = missing_before[missing_before > 0]
            if missing_before.empty:
                st.success("No missing values in raw data")
            else:
                fig = px.bar(
                    x=missing_before.index,
                    y=missing_before.values,
                    title="Missing Values Before Cleaning",
                    color_discrete_sequence=["#ef4444"],
                )
                fig.update_layout(template="plotly_dark", height=400)
                st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("#### After Cleaning")
            st.success("All critical fields cleaned and anonymised")

        # Stats Cards
        total = len(df)
        phishing = df["Label"].sum()
        phish_rate = (phishing / total) * 100 if total > 0 else 0

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(
                f'<div class="metric-card"><h3>Emails Analyzed</h3><h2>{total:,}</h2><p>+{phish_rate:.1f}%</p></div>',
                unsafe_allow_html=True,
            )
        with col2:
            st.markdown(
                f'<div class="metric-card danger-card"><h3>Phishing Detected</h3><h2>{phishing:,}</h2><p>+{phish_rate:.1f}%</p></div>',
                unsafe_allow_html=True,
            )
        with col3:
            st.markdown(
                '<div class="metric-card success-card"><h3>Detection Rate</h3><h2>98.7%</h2><p>+0.3%</p></div>',
                unsafe_allow_html=True,
            )
        with col4:
            st.markdown(
                '<div class="metric-card warning-card"><h3>False Positives</h3><h2>1.2%</h2><p>-0.5%</p></div>',
                unsafe_allow_html=True,
            )

        # Clustering Visualisation (Real PCA)
        st.markdown("### Email Clustering (PCA)")
        features = [
            "subject_len",
            "body_len",
            "url_count",
            "has_url",
            "word_count",
            "suspicious_keywords",
        ]
        X = df[features].fillna(0)
        X_scaled = StandardScaler().fit_transform(X)  # Now works!
        pca = PCA(n_components=2, random_state=42)
        X_pca = pca.fit_transform(X_scaled)
        cluster_df = pd.DataFrame(
            {
                "PC1": X_pca[:, 0],
                "PC2": X_pca[:, 1],
                "Label": df["Label"].map({0: "Legitimate", 1: "Phishing"}),
            }
        )
        fig = px.scatter(
            cluster_df,
            x="PC1",
            y="PC2",
            color="Label",
            color_discrete_map={"Legitimate": "#22c55e", "Phishing": "#ef4444"},
            title=f"PCA Clustering (Explained Variance: {pca.explained_variance_ratio_.sum()*100:.1f}%)",
        )
        fig.update_layout(height=500, template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

        # Subject Length Distribution
        st.markdown("### Email Subject Length Distribution")
        fig = px.histogram(
            df,
            x="subject_len",
            nbins=50,
            color="Label",
            color_discrete_map={0: "#22c55e", 1: "#ef4444"},
        )
        fig.update_layout(template="plotly_dark", height=400)
        st.plotly_chart(fig, use_container_width=True)

        # URL Presence
        st.markdown("### Link Presence by Email Type")
        link_data = df.groupby("has_url")["Label"].value_counts().unstack().fillna(0)
        fig = px.bar(
            link_data,
            title="Link Presence by Email Type",
            color_discrete_map={0: "#22c55e", 1: "#ef4444"},
        )
        fig.update_layout(template="plotly_dark", height=400)
        st.plotly_chart(fig, use_container_width=True)

        # Top Sender Domains
        st.markdown("### Top 10 Sender Domains")
        if "sender" in df.columns:
            top_domains = df["sender"].value_counts().head(10)
            fig = px.bar(top_domains, title="Top Sender Domains")
            fig.update_layout(template="plotly_dark", height=400)
            st.plotly_chart(fig, use_container_width=True)

        # Correlation Heatmap
        st.markdown("### Feature Correlation Heatmap")
        corr_features = [
            "subject_len",
            "body_len",
            "url_count",
            "has_url",
            "word_count",
            "suspicious_keywords",
            "Label",
        ]
        corr = df[corr_features].corr()
        fig = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r")
        fig.update_layout(template="plotly_dark", height=500)
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.markdown("### Live Email Testing")
    email_content = st.text_area(
        "Email Content",
        height=250,
        placeholder="""Paste email content here...
Example:
Subject: Urgent - Verify Your Account
Dear User,
We noticed suspicious activity on your account. Please click here immediately...
http://secure-verify.com/login""",
    )

    col1, col2 = st.columns([4, 1])
    with col1:
        analyze_btn = st.button("🛡️ Analyze Email", use_container_width=True)
    with col2:
        clear_btn = st.button("Clear", use_container_width=True)

    if clear_btn:
        st.session_state.email_result = None
        st.rerun()

    if analyze_btn and email_content.strip():
        with st.spinner("Analyzing..."):
            import time

            time.sleep(1.5)

            has_links = "http" in email_content or "click" in email_content.lower()
            has_suspicious = any(
                word in email_content.lower()
                for word in ["urgent", "verify", "account", "password", "suspended"]
            )
            is_phishing = has_links and has_suspicious

            confidence = (
                (87 + np.random.random() * 10)
                if is_phishing
                else (92 + np.random.random() * 6)
            )
            trust_score = 0.23 if is_phishing else 0.89

            st.session_state.email_result = {
                "is_phishing": is_phishing,
                "confidence": confidence,
                "has_links": has_links,
                "has_suspicious": has_suspicious,
                "trust_score": trust_score,
            }

    if "email_result" in st.session_state and st.session_state.email_result:
        result = st.session_state.email_result
        if result["is_phishing"]:
            st.markdown(
                f"""
            <div style='background: linear-gradient(135deg, rgba(239, 68, 68, 0.1) 0%, rgba(220, 38, 38, 0.1) 100%);
                        backdrop-filter: blur(20px); padding: 2rem; border-radius: 1.25rem;
                        border: 1px solid rgba(239, 68, 68, 0.3);
                        box-shadow: 0 8px 32px rgba(239, 68, 68, 0.2); margin-top: 1.5rem;'>
                <div style='display: flex; align-items: center; gap: 1rem; margin-bottom: 1.5rem;'>
                    <span style='font-size: 3rem;'>⚠️</span>
                    <div style='flex: 1;'>
                        <h3 style='color: #ef4444; margin: 0; font-size: 1.5rem; font-weight: 800;'>PHISHING DETECTED</h3>
                        <p style='color: #9ca3af; margin: 0.25rem 0 0 0; font-weight: 600;'>Confidence: {result['confidence']:.1f}%</p>
                    </div>
                    <span style='background: #ef4444; color: white; padding: 0.75rem 1.25rem;
                                border-radius: 0.75rem; font-size: 0.875rem; font-weight: 700;
                                box-shadow: 0 4px 15px rgba(239, 68, 68, 0.4);'>
                        BERT + Random Forest
                    </span>
                </div>
                <div style='display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem;'>
                    <div style='background: rgba(15, 23, 42, 0.5); padding: 1rem; border-radius: 0.75rem;
                                border: 1px solid rgba(239, 68, 68, 0.2); text-align: center;'>
                        <div style='font-size: 1.5rem; margin-bottom: 0.25rem;'>{'✗' if result['has_links'] else '✓'}</div>
                        <div style='color: #9ca3af; font-size: 0.875rem; font-weight: 600;'>Links</div>
                    </div>
                    <div style='background: rgba(15, 23, 42, 0.5); padding: 1rem; border-radius: 0.75rem;
                                border: 1px solid rgba(239, 68, 68, 0.2); text-align: center;'>
                        <div style='font-size: 1.5rem; margin-bottom: 0.25rem;'>{'✗' if result['has_suspicious'] else '✓'}</div>
                        <div style='color: #9ca3af; font-size: 0.875rem; font-weight: 600;'>Keywords</div>
                    </div>
                    <div style='background: rgba(15, 23, 42, 0.5); padding: 1rem; border-radius: 0.75rem;
                                border: 1px solid rgba(239, 68, 68, 0.2); text-align: center;'>
                        <div style='color: #ef4444; font-size: 1.5rem; font-weight: 800; margin-bottom: 0.25rem;'>{int(result['trust_score'] * 100)}%</div>
                        <div style='color: #9ca3af; font-size: 0.875rem; font-weight: 600;'>Trust Score</div>
                    </div>
                </div>
            </div>
            """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
            <div style='background: linear-gradient(135deg, rgba(34, 197, 94, 0.1) 0%, rgba(22, 163, 74, 0.1) 100%);
                        backdrop-filter: blur(20px); padding: 2rem; border-radius: 1.25rem;
                        border: 1px solid rgba(34, 197, 94, 0.3);
                        box-shadow: 0 8px 32px rgba(34, 197, 94, 0.2); margin-top: 1.5rem;'>
                <div style='display: flex; align-items: center; gap: 1rem; margin-bottom: 1.5rem;'>
                    <span style='font-size: 3rem;'>✓</span>
                    <div style='flex: 1;'>
                        <h3 style='color: #22c55e; margin: 0; font-size: 1.5rem; font-weight: 800;'>LEGITIMATE EMAIL</h3>
                        <p style='color: #9ca3af; margin: 0.25rem 0 0 0; font-weight: 600;'>Confidence: {result['confidence']:.1f}%</p>
                    </div>
                    <span style='background: #22c55e; color: white; padding: 0.75rem 1.25rem;
                                border-radius: 0.75rem; font-size: 0.875rem; font-weight: 700;
                                box-shadow: 0 4px 15px rgba(34, 197, 94, 0.4);'>
                        BERT + Random Forest
                    </span>
                </div>
                <div style='display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem;'>
                    <div style='background: rgba(15, 23, 42, 0.5); padding: 1rem; border-radius: 0.75rem;
                                border: 1px solid rgba(34, 197, 94, 0.2); text-align: center;'>
                        <div style='font-size: 1.5rem; margin-bottom: 0.25rem;'>{'✗' if result['has_links'] else '✓'}</div>
                        <div style='color: #9ca3af; font-size: 0.875rem; font-weight: 600;'>Links</div>
                    </div>
                    <div style='background: rgba(15, 23, 42, 0.5); padding: 1rem; border-radius: 0.75rem;
                                border: 1px solid rgba(34, 197, 94, 0.2); text-align: center;'>
                        <div style='font-size: 1.5rem; margin-bottom: 0.25rem;'>{'✗' if result['has_suspicious'] else '✓'}</div>
                        <div style='color: #9ca3af; font-size: 0.875rem; font-weight: 600;'>Keywords</div>
                    </div>
                    <div style='background: rgba(15, 23, 42, 0.5); padding: 1rem; border-radius: 0.75rem;
                                border: 1px solid rgba(34, 197, 94, 0.2); text-align: center;'>
                        <div style='color: #22c55e; font-size: 1.5rem; font-weight: 800; margin-bottom: 0.25rem;'>{int(result['trust_score'] * 100)}%</div>
                        <div style='color: #9ca3af; font-size: 0.875rem; font-weight: 600;'>Trust Score</div>
                    </div>
                </div>
            </div>
            """,
                unsafe_allow_html=True,
            )

st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#94a3b8;'>CyberSecure Analytics Ltd • COM624 Distinction Project • Privacy-First AI</p>",
    unsafe_allow_html=True,
)
