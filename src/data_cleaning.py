import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Load raw and Overview
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


# Strip whitespace and Standardize column
def clean_whitespace(text):
    if pd.isna(text):
        return text
    return re.sub(r"\s+", " ", text).strip()


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
    text = re.sub(r"\S+@\S+", "<EMAIL>", text)
    text = re.sub(r"http\S+|www\.\S+", "<LINK>", text)
    text = re.sub(r"(\+?\d[\d\-\(\) ]{7,}\d)", "<PHONE>", text)
    text = re.sub(r"\d+", "<NUMBER>", text)
    text = re.sub(r"[^a-z\s<>]", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


# Created a copy of the dataset for cleaning
df_clean = df_raw.copy()

# Handled missing values
# Fill sender and receiver with 'Unknown'
for col in ["sender", "receiver"]:
    if col not in df_clean.columns:
        df_clean[col] = df_raw[col].fillna("Unknown")


# Fill missing values in subject, or body if empty
for col in ["subject", "body"]:
    if col not in df_clean.columns:
        df_clean[col] = df_raw[col].fillna("Unknown")
