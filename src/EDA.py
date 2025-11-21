import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Load the clean dataset and overviewd
df = pd.read_csv("../data/processed/Nazario_cleanedData.csv")

print(df.info())
print(df.isna().sum())
print(df.shape)
