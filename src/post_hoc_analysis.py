import statsmodels.api as sa
import statsmodels.formula.api as sfa
import scikit_posthocs as sp
import numpy as np
import pandas as pd
import scipy.stats as ss
import matplotlib.pyplot as plt
from src.data_utils import encode_data

from statsmodels.datasets.utils import Dataset

# https://scikit-posthocs.readthedocs.io/en/latest/tutorial/#non-parametric-anova-with-post-hoc-tests
def post_hoc_analysis_short(data = None, df = None):
    """
    https://scikit-posthocs.readthedocs.io/en/latest/tutorial/#parametric-anova-with-post-hoc-tests
    https://scikit-posthocs.readthedocs.io/en/latest/tutorial/#non-parametric-anova-with-post-hoc-tests
    """
    x = data
    val_col = "Temperature____"
    group_col = "Humidity"
    lm = sfa.ols(f"{val_col} ~ {group_col}", data=df).fit()
    anova = sa.stats.anova_lm(lm)
    print(anova)
    posthoc_ttest = sp.posthoc_ttest(df, val_col=val_col, group_col=group_col, p_adjust='holm')
    print("posthoc_ttest", posthoc_ttest)
    print(data)
    posthoc_conover_results = sp.posthoc_conover(df, val_col=val_col, group_col=group_col, p_adjust='holm')
    print("posthoc conover results", posthoc_conover_results)


def post_hoc_analysis_long(data = None, df = None):
    """
    https://scikit-posthocs.readthedocs.io/en/latest/tutorial/#parametric-anova-with-post-hoc-tests
    https://scikit-posthocs.readthedocs.io/en/latest/tutorial/#non-parametric-anova-with-post-hoc-tests
    """
    x = data
    val_col = "Temperature____"
    group_col = "Humidity"

    lm = sfa.ols("Temperature____ ~ Humidity", data=df).fit()
    anova = sa.stats.anova_lm(lm)
    print(anova)
    sp.posthoc_ttest(df, val_col=val_col, group_col=group_col, p_adjust='holm')

    data = [df.loc[ids, val_col].values for ids in df.groupby(group_col).groups.values()]
    H, p = ss.kruskal(*data)
    print("kruskal results: ", H, p)

    pc = sp.posthoc_ttest(df, val_col=val_col, group_col=group_col)
    heatmap_args = {'linewidths': 0.25, 'linecolor': '0.5', 'clip_on': False, 'square': True, 'cbar_ax_bbox': [0.80, 0.35, 0.04, 0.3]}
    sp.sign_plot(pc, **heatmap_args)
    plt.savefig("data/images/posthoc_ttest_3.png")
    plt.show()

    pc = sp.posthoc_conover(df, val_col=val_col, group_col=group_col)
    cmap = ['1', '#fb6a4a', '#08306b', '#4292c6', '#c6dbef']
    heatmap_args = {'cmap': cmap, 'linewidths': 0.25, 'linecolor': '0.5', 'clip_on': False, 'square': True, 'cbar_ax_bbox': [0.80, 0.35, 0.04, 0.3]}
    sp.sign_plot(pc, **heatmap_args)
    plt.savefig("data/images/posthoc_conover_4.png")
    plt.show()



def main():
    post_hoc_analysis_short()
    post_hoc_analysis_long()

if __name__ == "__main__":
    main()
