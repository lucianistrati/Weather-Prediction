from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
from src.utils import get_regress_perf_metrics, plot_actual_and_predicted_feature
from src.data_utils import encode_data
import statsmodels.api as sm
from typing import Optional


def sk_linear_regression(X_train, X_test, y_train, y_test):
    # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
    model_name = "sk linear regression"
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)

    logging_metrics_list = get_regress_perf_metrics(y_test,
                                                    y_pred,
                                                    model_name)

    print(model_name, logging_metrics_list)

    plot_actual_and_predicted_feature(y_test,
                                      y_pred,
                                      model_name)

    return logging_metrics_list, model_name


def ols_linear_regression(X, y):
    model_name = "ols linear regression"
    X = sm.add_constant(X)
    model = sm.OLS(y, X)
    results = model.fit()

    y_pred = model.predict(params=results.params)

    logging_metrics_list = get_regress_perf_metrics(y,
                                                    y_pred,
                                                    model_name)

    print(model_name, logging_metrics_list)

    plot_actual_and_predicted_feature(y,
                                      y_pred,
                                      model_name)

    return logging_metrics_list, model_name
