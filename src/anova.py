from scipy.stats import ttest_ind, kstest, normaltest, f_oneway, ttest_1samp
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
import statsmodels.api as sa
import statsmodels.formula.api as sfa
import scikit_posthocs as sp
import scipy.stats as ss




def one_way_anova(data):
    return f_oneway(data)

def two_way_anova(df):

    # create data
    df = pd.DataFrame({'water': np.repeat(['daily', 'weekly'], 15),
                       'sun': np.tile(np.repeat(['low', 'med', 'high'], 5), 2),
                       'height': [6, 6, 6, 5, 6, 5, 5, 6, 4, 5,
                                  6, 6, 7, 8, 7, 3, 4, 4, 4, 5,
                                  4, 4, 4, 4, 4, 5, 6, 6, 7, 8]})


    # perform two-way ANOVA
    model = ols('height ~ C(water) + C(sun) + C(water):C(sun)', data=df).fit()
    sm.stats.anova_lm(model, typ=2)

# https://scikit-posthocs.readthedocs.io/en/latest/tutorial/#non-parametric-anova-with-post-hoc-tests
def post_hoc_analysis(data):#???
    x = data #???
    df = sa.datasets.get_rdataset('iris').data

    lm = sfa.ols('Sepal.Width ~ C(Species)', data=df).fit()
    anova = sa.stats.anova_lm(lm)
    print(anova)
    sp.posthoc_ttest(df, val_col='Sepal.Width', group_col='Species', p_adjust='holm')
    H, p = ss.kruskal(*data)
    sp.posthoc_conover(df, val_col='Sepal.Width', group_col='Species', p_adjust='holm')

    ss.friedmanchisquare(*data.T)
    sp.posthoc_nemenyi_friedman(data)
    sp.posthoc_nemenyi_friedman(data, y_col='y', block_col='blocks', group_col='groups', melted=True)
    sp.posthoc_conover(x, p_adjust='holm')

    df = sa.datasets.get_rdataset('iris').data
    sp.posthoc_conover(df, val_col='Sepal.Width', group_col='Species', p_adjust='holm')

    pc = sp.posthoc_conover(x, val_col='values', group_col='groups')
    heatmap_args = {'linewidths': 0.25, 'linecolor': '0.5', 'clip_on': False, 'square': True, 'cbar_ax_bbox': [0.80, 0.35, 0.04, 0.3]}
    sp.sign_plot(pc, **heatmap_args)

    pc = sp.posthoc_conover(x, val_col='values', group_col='groups')
    # Format: diagonal, non-significant, p<0.001, p<0.01, p<0.05
    cmap = ['1', '#fb6a4a', '#08306b', '#4292c6', '#c6dbef']
    heatmap_args = {'cmap': cmap, 'linewidths': 0.25, 'linecolor': '0.5', 'clip_on': False, 'square': True, 'cbar_ax_bbox': [0.80, 0.35, 0.04, 0.3]}
    sp.sign_plot(pc, **heatmap_args)