from scipy.stats import ttest_ind, kstest, normaltest, f_oneway, ttest_1samp
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
import statsmodels.api as sa
import statsmodels.formula.api as sfa
import scikit_posthocs as sp
import scipy.stats as ss
import matplotlib.pyplot as plt
import pingouin as pg


def two_way_anova(df = None):
    """
    https://www.statology.org/two-way-anova-python/
    https://www.geeksforgeeks.org/how-to-perform-a-two-way-anova-in-python/
    https://www.kaggle.com/code/alexmaszanski/two-way-anova-with-python/notebook
    https://www.marsja.se/three-ways-to-carry-out-2-way-anova-with-python/
    https://www.reneshbedre.com/blog/anova.html
    """
    pred_var = "Temperature____"
    var_1 = "Humidity"
    var_2 = "Pressure__millibars_"

    model = ols(f'{pred_var} ~ {var_1} + {var_2}', data=df).fit()
    anova_type = 2
    anova_results = sm.stats.anova_lm(model, typ=anova_type)
    print("anova results", anova_results)
    fit = True
    line = "45"

    residuals = model.resid
    fig = sm.qqplot(residuals, ss.t, fit=fit, line=line)
    plt.savefig("data/images/residuals_two_way_anova_task_1.png")
    plt.show()

    do_detail = True

    dependent_variable = "Temperature____"
    between_variables = ["Humidity", "Pressure__millibars_"]
    anova_results = pg.anova(df, dv=dependent_variable, between=between_variables, detailed=do_detail)
    print("anova results: ", anova_results)
