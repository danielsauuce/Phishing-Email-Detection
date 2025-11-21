import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Load raw and Overview
df = pd.read_csv(
    "../data/raw/Nazario-5.csv",
    encoding="ISO-8859-1",
    na_values=["NA", "na", "Unknown", ""],
)

print(df.head())
print(df.info())
print(df.shape)
print(df.isna().sum())
print(df.describe())


# Cleaning functions
# Remove HTML tags
def remove_html(text):
    if pd.isna(text):
        return text
    return re.sub(r"<.*?>", "", text)


# Extract URLS
def extract_urls(text):
    if pd.isna(text):
        return []
    url_pattern = r"(https?://[^\s]+)"
    return re.findall(url_pattern, text)


# Strip whitespace
def clean_whitespace(text):
    if pd.isna(text):
        return text
    return re.sub(r"\s+", " ", text).strip()


# Remove non-ASCII characters
def clean_non_ascii(text):
    if pd.isna(text):
        return text
    return text.encode("ascii", errors="ignore").decode()


