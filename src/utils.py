from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.utils import check_array
from src.logger import empty_regress_loggings
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.metrics import matthews_corrcoef

EPSILON = 1e-10

def mean_absolute_percentage_error(y_true, y_pred):
    y_true = check_array(y_true)
    y_pred = check_array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100



def _error(actual: np.ndarray, predicted: np.ndarray):
    return actual - predicted


def _percentage_error(actual: np.ndarray, predicted: np.ndarray):
    return _error(actual, predicted) / (actual + EPSILON)

def mape(actual: np.ndarray, predicted: np.ndarray):
    return np.mean(np.abs(_percentage_error(actual, predicted)))

def mda(actual: np.ndarray, predicted: np.ndarray):
    """ Mean Directional Accuracy """
    return np.mean((np.sign(actual[1:] - actual[:-1]) == np.sign(predicted[1:] - predicted[:-1])).astype(int))


def get_regress_perf_metrics(y_test, y_pred, model_name="",
                             target_feature="",
                             logging_metrics_list=empty_regress_loggings(),
                             visualize_metrics=False):
    if visualize_metrics:
        print("For " + model_name + " regression algorithm the following "
                                    "performance metrics were determined:")

    if target_feature == 'all':
        y_test = y_test.flatten()
        y_pred = y_pred.flatten()

    for i in range(len(logging_metrics_list)):
        if logging_metrics_list[i][0] == "MSE":
            logging_metrics_list[i][1] = str(mean_squared_error(y_test, y_pred))
        elif logging_metrics_list[i][0] == "MAE":
            logging_metrics_list[i][1] = str(mean_absolute_error(y_test,
                                                                 y_pred))
        elif logging_metrics_list[i][0] == "R2":
            logging_metrics_list[i][1] = str(r2_score(y_test, y_pred))
        elif logging_metrics_list[i][0] == "MAPE":
            logging_metrics_list[i][1] = str(mape(y_test, y_pred))
        elif logging_metrics_list[i][0] == "MDA":
            logging_metrics_list[i][1] = str(mda(y_test, y_pred))
        elif logging_metrics_list[i][0] == "MAD":
            logging_metrics_list[i][1] = 0.0

    if visualize_metrics:
        print("MSE: ", mean_squared_error(y_test, y_pred))
        print("MAE: ", mean_absolute_error(y_test, y_pred))
        print("R squared score: ", r2_score(y_test, y_pred))
        print("Mean absolute percentage error:", mape(y_test, y_pred))
        try:
            print("Mean directional accuracy:", mda(y_test, y_pred))
        except TypeError:
            print("Type error", model_name)

    return logging_metrics_list


import matplotlib.pyplot as plt
def plot_actual_and_predicted_feature(actual_price, predicted_price, model_name,
                                    date_interval=None, preview=True):
    if not date_interval:
        date_interval = range(1, len(actual_price) + 1)
    plt.plot(date_interval, actual_price, "g", label = "Actual feature")
    plt.plot(date_interval, predicted_price, "b", label = "Predicted feature")
    plt.legend()
    plt.title(model_name)
    plt.savefig(f"data/images/{model_name}.png")
    if preview:
        plt.show()


from scipy.stats import ttest_ind, kstest, normaltest, f_oneway, ttest_1samp
import numpy as np
import pandas as pd

from statsmodels.tsa.stattools import adfuller
from scipy.stats import permutation_test

def statistic(x, y, axis):
    return np.mean(x, axis=axis) - np.mean(y, axis=axis)

def permutation_test_fn(X, y):
    res = permutation_test((X, y), statistic)
    return res.statistic, res.p_value

def normality_test(data):
    return normaltest(data)

def two_sample_t_test(a, b):
    return ttest_ind(a, b)

def ks_test(a, b):
    return kstest(a, b)

def quantiles(data, q:int=50):
    return np.quantile(data, q)

def one_sample_t_test(data, popmean: float=0.0):
    return ttest_1samp(data, popmean)

def one_way_anova(data):
    return f_oneway(data)



def is_data_stationary(time_series):
    result = adfuller(time_series)
    print('Augmented Dickey-Fuller Test:')

    if result[1] <= 0.05:
        return True
    else:
        return False

def transform_data_to_stationary(df, target_feature):
    max_shift_counter = 101
    for shift_counter in range(max_shift_counter):
        diff_arr = df[target_feature] - df[target_feature].shift(shift_counter)
        diff_arr = diff_arr.dropna()
        if is_data_stationary(diff_arr):
            df[target_feature] = diff_arr
            break
    return df[target_feature]



def main():
    a = np.random.rand(1, 100)
    b = np.random.rand(1, 100)

    logging_metrics_list = get_regress_perf_metrics(a, b)

    print(logging_metrics_list)

if __name__=='__main__':
    main()

