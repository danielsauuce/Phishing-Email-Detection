import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Load the cleaned Dataset
df = pd.read_csv("../data/processed/Nazario_cleanedData.csv")

# Load engineered feature dataset
df_fe = pd.read_csv("../data/engineered/featuure_engineering_ready.csv")


# Drop rows with missing values
# df.dropna(inplace=True)
print(df.shape)

#  Convert label to descriptive names
df["label_name"] = df["label"].map({0: "Legitimate", 1: "Phishing"})
print(df["label_name"].value_counts())


# Distribution of Email Types
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x="label_name", order=["Legitimate", "Phishing"])
plt.title("Legitimate vs Phishing Emails")
plt.xlabel("Email Type")
plt.ylabel("Count")

# Add percentages on top
total = len(df)
for p in plt.gca().patches:
    height = p.get_height()
    plt.text(
        p.get_x() + p.get_width() / 2.0,
        height + 10,
        ha="center",
        fontsize=12,
    )
plt.tight_layout()
plt.show()


# URL Count Analysis/ Distribution
plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
sns.boxplot(data=df, x="label_name", y="url_count")
plt.title("URL Count by Email Type")
plt.ylabel("Number of URLs")

plt.subplot(1, 2, 2)
sns.histplot(
    data=df[df["url_count"] < 20], x="url_count", hue="label_name", bins=20, alpha=0.7
)
plt.title("Distribution of URL Count (zoom < 20)")
plt.xlabel("Number of URLs")
plt.tight_layout()
plt.show()

print("\nURL Count Stats:")
print(df.groupby("label_name")["url_count"].describe())


# Email Length Features
plt.figure(figsize=(14, 10))

plt.subplot(2, 2, 1)
sns.boxplot(data=df, x="label_name", y="subject_length")
plt.title("Subject Length by Email Type")

plt.subplot(2, 2, 2)
sns.histplot(data=df, x="subject_length", hue="label_name", bins=50, alpha=0.7)
plt.xlim(0, 150)
plt.title("Subject Length Distribution")

plt.subplot(2, 2, 3)
sns.boxplot(data=df, x="label_name", y="body_length")
plt.title("Body Length by Email Type")

plt.subplot(2, 2, 4)
sns.histplot(
    data=df[df["body_length"] < 5000],
    x="body_length",
    hue="label_name",
    bins=50,
    alpha=0.7,
)
plt.title("Body Length Distribution (truncated)")
plt.tight_layout()
plt.show()


# Suspicious Keywords in Subject
keywords = [
    "urgent",
    "account",
    "verify",
    "password",
    "security",
    "login",
    "bank",
    "paypal",
    "confirm",
    "update",
    "suspended",
    "immediate",
]

keyword_count = []
for word in keywords:
    count = df["subject_clean"].str.contains(word, case=False, na=False).sum()
    phishing_rate = df[df["subject_clean"].str.contains(word, case=False, na=False)][
        "label"
    ].mean()
    keyword_count.append(
        {"keyword": word, "count": count, "phishing_rate": phishing_rate}
    )

keyword_df = pd.DataFrame(keyword_count).sort_values("count", ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(data=keyword_df, y="keyword", x="count")
plt.title("Frequency of Suspicious Keywords in Subject")
plt.xlabel("Number of Emails Containing Keyword")
plt.show()

# print("\nPhishing Rate When Keyword is in Subject:")
# for _, row in keyword_df.iterrows():
#     if row["count"] > 0:
#         print(
#             f"{row['keyword']:12}: {row['count']:4} emails → {row['phishing_rate']:.3f} phishing rate"
#         )


# Sender & Receiver Domain Analysis
plt.figure(figsize=(15, 10))

# Top 10 sender domains overall
plt.subplot(2, 1, 1)
top_senders = df["sender_domain"].value_counts().head(10)
sns.barplot(x=top_senders.values, y=top_senders.index)
plt.title("Top 10 Sender Domains (Overall)")
plt.xlabel("Number of Emails")

# Phishing rate per domain (domains with >= 10 emails)
plt.subplot(2, 1, 2)
domain_stats = df.groupby("sender_domain")["label"].agg(["count", "mean"])
domain_stats = (
    domain_stats[domain_stats["count"] >= 10]
    .sort_values("mean", ascending=False)
    .head(15)
)
sns.barplot(data=domain_stats.reset_index(), y="sender_domain", x="mean", orient="h")
plt.title("Top 15 Most Suspicious Sender Domains (min 10 emails)")
plt.xlabel("Phishing Rate (0=all phishing, 1=all legitimate)")
plt.ylabel("Sender Domain")
plt.tight_layout()
plt.show()

# Receiver domain analysis
plt.figure(figsize=(15, 10))

plt.subplot(2, 1, 1)
top_receivers = df["receiver_domain"].value_counts().head(10)
sns.barplot(x=top_receivers.values, y=top_receivers.index)
plt.title("Top 10 Receiver Domains (Overall)")
plt.xlabel("Number of Emails")

plt.subplot(2, 1, 2)
receiver_stats = df.groupby("receiver_domain")["label"].agg(["count", "mean"])
receiver_stats = (
    receiver_stats[receiver_stats["count"] >= 10]
    .sort_values("mean", ascending=False)
    .head(15)
)
sns.barplot(
    data=receiver_stats.reset_index(), y="receiver_domain", x="mean", orient="h"
)
plt.title("Top 15 Most Suspicious Receiver Domains (min 10 emails)")
plt.xlabel("Phishing Rate (0=all phishing, 1=all legitimate)")
plt.ylabel("Receiver Domain")
plt.tight_layout()
plt.show()


# Correlation Heatmap (Numerical Features)
plt.figure(figsize=(8, 6))
numerical = ["url_count", "subject_length", "body_length", "num_recipients", "label"]
corr = df[numerical].corr()
sns.heatmap(corr, annot=True, cmap="coolwarm", center=0, square=True)
plt.title("Correlation Matrix of Numerical Features")
plt.tight_layout()
plt.show()


# Datetime Feature Analysis
plt.figure(figsize=(18, 5))

# Activity by Hour
plt.subplot(1, 3, 1)
sns.countplot(data=df, x="hour", hue="label_name", palette="viridis")
plt.title("Email Activity by Hour of Day")
plt.xlabel("Hour (0=Midnight, 23=11 PM)")

# Activity by Day of Week
plt.subplot(1, 3, 2)
sns.countplot(data=df, x="day_of_week", hue="label_name", palette="viridis")
plt.title("Email Activity by Day of Week")
plt.xlabel("Day of Week (0=Monday, 6=Sunday)")

# Business Hours vs. Off-Hours
plt.subplot(1, 3, 3)
sns.countplot(data=df, x="is_business_hours", hue="label_name", palette="viridis")
plt.xticks([0, 1], ["Off-Hours (18-8)", "Business Hours (9-17)"])
plt.title("Activity: Business vs. Off-Hours")

plt.tight_layout()
plt.show()


