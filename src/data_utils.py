import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder


def load_data():
    df = pd.read_csv("data/weatherHistory.csv")
    return df


def encode_data(df):
    cols = ["Formatted_Date", "Summary", "Precip_Type", "Daily_Summary"]
    for col in cols:
        le = LabelEncoder()
        values = df[col].to_list()
        encoded_values = le.fit_transform(values)
        print(encoded_values.dtype)
        df[col] = encoded_values

    return df


def main():
    df = load_data()

    days = list()
    datetimes = df["Formatted Date"].to_list()
    for datetime in datetimes:
        days.append(datetime.split()[0])

    week = df.iloc[24: 24 * 8]
    week.to_csv("data/first_week_24_hours.csv")

    week = df.iloc[24: 24 * 8:24]
    week.to_csv("data/first_week_midnight.csv")

    year = df.iloc[::24]
    year.to_csv("data/weatherHistory_midnight.csv")

    days = list(set(days))
    data = encode_data(df)


def first_week():
    df = pd.read_csv("data/first_week_24_hours_underlined_columns.csv")
    df = encode_data(df)

    df[["Pressure__millibars_"]].plot()
    plt.savefig("data/images/1.png")
    plt.show()

    df[["Humidity"]].plot()
    plt.savefig("data/images/2.png")
    plt.show()

    df[["Temperature____"]].plot()
    plt.savefig("data/images/3.png")
    plt.show()


if __name__ == "__main__":
    first_week()
