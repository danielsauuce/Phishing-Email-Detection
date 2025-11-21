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
