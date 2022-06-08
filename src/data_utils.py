import numpy as np
import pandas as pd

def load_data():
    df = pd.read_csv("data/weatherHistory.csv")
    return df

df = load_data()

print(df.describe())
print(df.columns)
print(df.head())
print(df.tail())
print(df.dtypes)
for col in df.columns.to_list():
    print(col, "-")
