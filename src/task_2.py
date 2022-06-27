import pandas as pd
from src.data_utils import encode_data
from src.goodness_of_fit_residual_analysis import goodness_of_fit_analysis
import numpy as np
from src.anova import two_way_anova
from src.utils import two_sample_t_test, permutation_test_fn, one_sample_t_test
from src.post_hoc_analysis import post_hoc_analysis_short, post_hoc_analysis_long
from src.linear_model import sk_linear_regression, ols_linear_regression
from sklearn.model_selection import train_test_split

def main():
    df = pd.read_csv("data/first_week_24_hours_underlined_columns.csv")
    df = encode_data(df)

    goodness_of_fit_analysis(df)

    g1 = df["Humidity"].to_numpy()
    g2 = df["Pressure__millibars_"].to_numpy()
    g3 = df["Temperature____"].to_numpy()


    data = np.r_[g1, g2, g3]

    two_way_anova(df)

    print("two sample t test: ", two_sample_t_test(g1, g2))
    print("permutation test: ", permutation_test_fn(g1, g2))
    print("one sample t test: ", one_sample_t_test(g1))

    post_hoc_analysis_short(data=data, df=df)
    post_hoc_analysis_long(data=data, df=df)

    pred_var = "Temperature____"
    var_1 = "Humidity"
    var_2 = "Pressure__millibars_"


    X = df[[var_1, var_2]].to_numpy()
    y = df[pred_var].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    sk_linear_regression(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
    ols_linear_regression(X=X, y=y)


if __name__ == "__main__":
    main()
