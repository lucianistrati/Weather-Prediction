import pandas as pd
from src.data_utils import encode_data
import matplotlib.pyplot as plt


def get_correlations():
    # https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.corr.html
    df = pd.read_csv("data/first_week_24_hours_underlined_columns.csv")
    df = encode_data(df)
    data = df.to_numpy()
    print(df.corr())
    print(type(df.corr()))
    print(df[df.columns].corr()['Temperature____'][:])

    pd.plotting.scatter_matrix(df, alpha=0.2)
    plt.show()


def main():
    get_correlations()


if __name__ == "__main__":
    main()
