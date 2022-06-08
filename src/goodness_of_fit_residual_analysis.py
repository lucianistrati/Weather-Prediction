import numpy as np
import pandas as pd


#import necessary libraries
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.formula.api import ols
# https://www.statology.org/residual-plot-python/
def goodness_of_fit_analysis(df):
    #fit simple linear regression model
    model = ols('rating ~ points', data=df).fit()

    #view model summary
    print(model.summary())


    #define figure size
    fig = plt.figure(figsize=(12,8))

    #produce regression plots
    fig = sm.graphics.plot_regress_exog(model, 'points', fig=fig)

    #fit multiple linear regression model
    model = ols('rating ~ assists + rebounds', data=df).fit()

    #view model summary
    print(model.summary())

    #create residual vs. predictor plot for 'assists'
    fig = plt.figure(figsize=(12,8))
    fig = sm.graphics.plot_regress_exog(model, 'assists', fig=fig)

    #create residual vs. predictor plot for 'assists'
    fig = plt.figure(figsize=(12,8))
    fig = sm.graphics.plot_regress_exog(model, 'rebounds', fig=fig)