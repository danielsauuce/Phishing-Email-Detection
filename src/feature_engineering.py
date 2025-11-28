import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
import warnings

warnings.filterwarnings("ignore")


# Load cleaned dataset
df = pd.read_csv("../data/processed/Nazario_cleanedData22.csv")
df.dropna(subset=["body_clean", "subject_clean"], inplace=True)


# Datetime features
if "date" in df.columns:
    df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True)
    df = df.dropna(subset=["date"]).reset_index(drop=True)
    df["hour"] = df["date"].dt.hour
    df["day_of_week"] = df["date"].dt.dayofweek
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    df["is_business_hours"] = df["hour"].between(9, 17).astype(int)


#  Metadata features
df["has_url"] = (df["url_count"] > 0).astype(int)
df["many_urls"] = (df["url_count"] > 3).astype(int)
df["no_url_but_link_token"] = (
    df["body_clean"].str.contains("<LINK>", case=False, na=False).astype(int)
)
df["has_phone"] = (
    df["body_clean"].str.contains("<PHONE>", case=False, na=False).astype(int)
)
df["has_email_token"] = (
    df["body_clean"].str.contains("<EMAIL>", case=False, na=False).astype(int)
)
df["body_subject_ratio"] = df["body_length"] / (df["subject_length"] + 1)
df["chars_per_url"] = df["body_length"] / (df["url_count"] + 1)
df["is_short_body"] = (df["body_length"] < 200).astype(int)
df["is_long_subject"] = (df["subject_length"] > 100).astype(int)


# Suspicious keyword flags
suspicious_keywords = [
    "urgent",
    "account",
    "verify",
    "suspend",
    "security",
    "password",
    "login",
    "bank",
    "paypal",
    "confirm",
    "update",
    "immediate",
    "action required",
    "limited time",
    "click here",
    "unusual activity",
    "payment",
    "invoice",
]

for word in suspicious_keywords:
    safe_col = word.replace(" ", "_")
    df[f"subj_has_{safe_col}"] = (
        df["subject_clean"].str.contains(word, case=False, na=False).astype(int)
    )
    df[f"body_has_{safe_col}"] = (
        df["body_clean"].str.contains(word, case=False, na=False).astype(int)
    )

keyword_cols = [
    c for c in df.columns if c.startswith("subj_has_") or c.startswith("body_has_")
]
df["total_suspicious_keywords"] = df[keyword_cols].sum(axis=1)


# Domain & sender features
sender_phish_score = df.groupby("sender_domain")["label"].mean()
receiver_phish_score = df.groupby("receiver_domain")["label"].mean()
df["sender_phish_score"] = df["sender_domain"].map(sender_phish_score).fillna(0.5)
df["receiver_phish_score"] = df["receiver_domain"].map(receiver_phish_score).fillna(0.5)
high_risk_senders = sender_phish_score[sender_phish_score < 0.3].index
df["from_high_risk_domain"] = df["sender_domain"].isin(high_risk_senders).astype(int)

non_reply_keywords = ["noreply", "support", "service", "alert"]
df["is_non_reply_sender"] = (
    df["sender"]
    .str.contains("|".join(non_reply_keywords), case=False, na=False, regex=True)
    .astype(int)
)
df["sender_user_len"] = df["sender"].apply(
    lambda x: len(str(x).split("@")[0]) if "@" in str(x) else 0
)

TOP_N = 50
top_senders = df["sender_domain"].value_counts().head(TOP_N).index.tolist()
df["sender_domain_grouped"] = df["sender_domain"].apply(
    lambda x: x if x in top_senders else "other_sender"
)
df["sender_domain_encoded"] = df["sender_domain_grouped"].factorize()[0]

top_receivers = df["receiver_domain"].value_counts().head(TOP_N).index.tolist()
df["receiver_domain_grouped"] = df["receiver_domain"].apply(
    lambda x: x if x in top_receivers else "other_receiver"
)
df["receiver_domain_encoded"] = df["receiver_domain_grouped"].factorize()[0]


# TF-IDF (subject + body)
df["full_text"] = df["subject_clean"].fillna("") + " " + df["body_clean"].fillna("")
tfidf = TfidfVectorizer(
    max_features=1000, ngram_range=(1, 2), min_df=3, max_df=0.9, stop_words="english"
)
tfidf_matrix = tfidf.fit_transform(df["full_text"])
tfidf_df = pd.DataFrame(
    tfidf_matrix.toarray(),
    index=df.index,
    columns=[f"tfidf_{i}" for i in range(tfidf_matrix.shape[1])],
)


# Scaling numeric/metadata features
metadata_features = [
    "url_count",
    "has_url",
    "many_urls",
    "no_url_but_link_token",
    "subject_length",
    "body_length",
    "body_subject_ratio",
    "chars_per_url",
    "num_recipients",
    "has_phone",
    "has_email_token",
    "is_short_body",
    "is_long_subject",
    "total_suspicious_keywords",
    "sender_phish_score",
    "receiver_phish_score",
    "from_high_risk_domain",
    "sender_user_len",
    "is_non_reply_sender",
    "sender_domain_encoded",
    "receiver_domain_encoded",
] + (
    ["hour", "day_of_week", "is_weekend", "is_business_hours"]
    if "hour" in df.columns
    else []
)

metadata_present = [c for c in metadata_features if c in df.columns]
scaler = StandardScaler()
df[metadata_present] = scaler.fit_transform(df[metadata_present])


# Dimensionality reduction for TF-IDF
tfidf_cols = [c for c in tfidf_df.columns]
svd = TruncatedSVD(n_components=100, random_state=42)
tfidf_reduced = svd.fit_transform(tfidf_df[tfidf_cols])
tfidf_reduced_df = pd.DataFrame(
    tfidf_reduced,
    columns=[f"svd_{i}" for i in range(tfidf_reduced.shape[1])],
    index=df.index,
)


# Combine metadata + reduced TF-IDF + label
X_final = pd.concat(
    [
        df[metadata_present].reset_index(drop=True),
        tfidf_reduced_df.reset_index(drop=True),
    ],
    axis=1,
)
X_final["label"] = df["label"].reset_index(drop=True)


# Save final CSV ready for clustering
output_dir = "../data/engineered"
os.makedirs(output_dir, exist_ok=True)
X_final.to_csv(os.path.join(output_dir, "featuure_engineering_ready.csv"), index=False)
