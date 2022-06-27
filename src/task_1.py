import pandas as pd
from src.data_utils import encode_data
from scipy.stats import bartlett
import numpy as np
from src.anova import two_way_anova
from src.utils import two_sample_t_test, permutation_test_fn, one_sample_t_test
from src.post_hoc_analysis import post_hoc_analysis_short, post_hoc_analysis_long
from statsmodels.datasets.utils import Dataset

def main():
    df = pd.read_csv("data/first_week_24_hours_underlined_columns.csv")
    print("Number of nan values: ", df.isna().sum())
    df = encode_data(df)
    data = df.to_numpy()
    print(data.shape)
    possible_feature_indexes = list(range(3, 12))
    feature_index = 3

    g1 = df["Humidity"].to_numpy()
    g2 = df["Pressure__millibars_"].to_numpy()
    g3 = df["Temperature____"].to_numpy()
    print(g1.shape, g2.shape, g3.shape)

    statistic, pvalue = bartlett(g1, g2, g3)

    print(f"bartlett results: {statistic, pvalue}")

    data = np.r_[g1, g2, g3]

    two_way_anova(df)

    print("two sample t test: ", two_sample_t_test(g1, g2))
    print("permutation test: ", permutation_test_fn(g1, g2))
    print("one sample t test: ", one_sample_t_test(g3))

    data = Dataset(data=df, title="Weather Dataset")
    post_hoc_analysis_short(data=data, df=df)
    post_hoc_analysis_long(data=data, df=df)



if __name__ == "__main__":
    main()
