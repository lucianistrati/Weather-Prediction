import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.formula.api import ols
from src.data_utils import encode_data


def goodness_of_fit_analysis(df = None):
    # https://www.statology.org/residual-plot-python/
    print(df.columns)
    model = ols('Temperature____ ~ Humidity', data=df).fit()

    print(model.summary())

    pred_var = "Temperature____"
    var_1 = "Humidity"
    var_2 = "Pressure__millibars_"

    model = ols(f'{pred_var} ~ {var_1} + {var_2}', data=df).fit()

    print(model.summary())

    fig = plt.figure(figsize=(12, 8))
    fig = sm.graphics.plot_regress_exog(model, f"{var_1}", fig=fig)
    plt.savefig(f"data/images/plot_regress_exog_{var_1}.png")
    plt.show()

    fig = plt.figure(figsize=(12, 8))
    fig = sm.graphics.plot_regress_exog(model, f"{var_2}", fig=fig)
    plt.savefig(f"data/images/plot_regress_exog_{var_2}.png")
    plt.show()
