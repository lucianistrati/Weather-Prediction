from sklearn.linear_model import LinearRegression
import numpy as np

from src.utils import get_regress_perf_metrics, plot_actual_and_predicted_feature

def linear_regression(X_train, y_train, y_test):
    model_name = "linear regression"
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    X_test, y_pred = [X_train[-1]], [y_train[-1]]

    for i in range(20):
        prediction = lr.predict(np.reshape(X_test[0], (1, 20)))
        X_test.append(X_test[-1][1:] + [prediction])
        y_pred.append(prediction)

    y_test = np.array(y_test)
    y_pred = np.array(y_pred)

    logging_metrics_list = get_regress_perf_metrics(y_test,
                                                    y_pred,
                                                    model_name)

    print(logging_metrics_list)
    plot_actual_and_predicted_feature(y_test,
                                    y_pred,
                                    model_name)

    return logging_metrics_list, model_name

