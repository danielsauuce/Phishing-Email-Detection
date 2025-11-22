import re
import pandas as pd
import matplotlib.pyplot as plt


# Load raw data and Overview
df_raw = pd.read_csv(
    "../data/raw/Nazario-5.csv",
    encoding="ISO-8859-1",
    na_values=["NA", "na", "Unknown", ""],
)

print(df_raw.head())
print(df_raw.info())
print(df_raw.shape)
print(df_raw.isna().sum())
print(df_raw.describe())


# Cleaning functions
# Extract URLS
def extract_urls(text):
    if pd.isna(text):
        return []
    url_pattern = r"(https?://[^\s]+)"
    return re.findall(url_pattern, text)


# """
#     Cleans and anonymises raw email text:
#     - Lowercase
#     - Replace emails with <EMAIL>
#     - Replace URLs with <LINK>
#     - Replace phone numbers with <PHONE>
#     - Replace numbers with <NUMBER>
#     - Remove special characters
#     """
def clean_and_anonymise_text(text):

    if pd.isna(text):
        return ""

    text = text.lower()

    # Replace email addresses
    text = re.sub(r"\S+@\S+", "<EMAIL>", text)

    # Replace URLs
    text = re.sub(r"http\S+|www\.\S+", "<LINK>", text)

    # Replace phone numbers (digits, +, -, parentheses, spaces)
    text = re.sub(r"\+?\d[\d\-\(\) ]{7,}\d", "<PHONE>", text)

    # Protect tokens before replacing numbers
    text = text.replace("<EMAIL>", "EMAILTOKEN")
    text = text.replace("<LINK>", "LINKTOKEN")
    text = text.replace("<PHONE>", "PHONETOKEN")

    # Replace remaining numbers
    # text = re.sub(r"\d+", "<NAME>", text)

    # Restore tokens
    text = text.replace("EMAILTOKEN", "<EMAIL>")
    text = text.replace("LINKTOKEN", "<LINK>")
    text = text.replace("PHONETOKEN", "<PHONE>")

    # Remove other non-alphabetic characters but keep < and >
    # text = re.sub(r"[^a-z\s<>]", "", text)

    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


# Created a copy of the dataset for cleaning
df_clean = df_raw.copy()

# Handled missing values
# Fill sender and receiver missing values
for col in ["sender", "receiver"]:
    if col in df_clean.columns:
        df_clean[col] = df_clean[col].fillna("Unknown")

# Fill missing values in subject and body
for col in ["subject", "body"]:
    if col in df_clean.columns:
        df_clean[col] = df_clean[col].fillna("Unknown")


# Handle missing dates using mode
if "date" in df_clean.columns:
    df_clean["date"] = pd.to_datetime(
        df_clean["date"], errors="coerce"
    )  # convert to datetime
    mode_date = df_clean["date"].mode(dropna=True)
    if not mode_date.empty:
        df_clean["date"] = df_clean["date"].fillna(mode_date[0])


# Overviewed to check the dataset as i proceed with the cleaning
print(df_clean.info())
print(df_clean.isna().sum())


# Applying the cleaning function and anonymisation
df_clean["subject_clean"] = df_clean["subject"].apply(clean_and_anonymise_text)
df_clean["body_clean"] = df_clean["body"].apply(clean_and_anonymise_text)

# Tokenise text by splitting on spaces
df_clean["subject_tokens"] = df_clean["subject_clean"].apply(lambda x: x.split())
df_clean["body_tokens"] = df_clean["body_clean"].apply(lambda x: x.split())

# Extract URLs and count
df_clean["urls"] = df_clean["body"].apply(extract_urls)
df_clean["url_count"] = df_clean["urls"].apply(len)

# Metadata features
df_clean["sender_domain"] = (
    df_clean["sender"].str.extract(r"@([\w\.-]+)")[0].fillna("Unknown")
)
df_clean["receiver_domain"] = (
    df_clean["receiver"].str.extract(r"@([\w\.-]+)")[0].fillna("Unknown")
)
df_clean["subject_length"] = df_clean["subject"].apply(len)
df_clean["body_length"] = df_clean["body"].apply(len)
df_clean["num_recipients"] = df_clean["receiver"].apply(
    lambda x: len(x.split(",")) if pd.notna(x) else 0
)
print(df_clean.info())  # dataset overview


# Visualisation of missing values
plt.figure(figsize=(10, 5))
# Before cleaning
df_raw.isna().sum().plot(kind="bar", color="red", alpha=0.7, label="Before Cleaning")
# After cleaning
df_clean.isna().sum().plot(kind="bar", color="green", alpha=0.7, label="After Cleaning")

# Comparison of missing values Before and After cleaning
plt.title("Missing Values Before vs After Cleaning")
plt.legend()
plt.tight_layout()
plt.show()


# Saved Cleaned Dataset
# Confirmed there is no missing values before saving
df_clean.isna().sum()
df_clean.to_csv("../data/processed/Nazario_cleanedData.csv", index=False)

# Preview the cleaned dataset
print("\nPreview of cleaned dataset:")
print(df_clean.head())
