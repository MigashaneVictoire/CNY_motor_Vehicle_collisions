from scipy import stats
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score

from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import f_regression 
from math import sqrt
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import math
import warnings

warnings.filterwarnings('ignore')
# set a default them for all my visuals
sns.set_theme(style="whitegrid")

def plot_residuals(actual, predicted):
    # substract the predicted from the actual to get the residual
    residual = predicted - actual

    # Plot the residual
    sns.regplot(x=np.arange(len(residual)),y=residual)
    fig = plt.gcf()
    return fig

def regression_errors(y, yhat):
    """
    parameters:
        y: actual target value
        yhat: predicted values
    return:
        Explained Variance and 
        (SSE, ESS, TSS, MSE, RMSE)
    """
    # create a datafame from the input values
    df_eval = pd.DataFrame({
    "tax_value": np.array(y),
    "y_hat":np.array(yhat)
    })

    # calculate predicted sum of squared error
    SSE = mean_squared_error(df_eval.tax_value, df_eval.y_hat) * len(df_eval)

    # compute explained sum of squares
    ESS = sum((df_eval.y_hat - df_eval.tax_value.mean())**2)

    # total sum of squares, mean squared 
    TSS = ESS + SSE

    # calculate root mean squared error
    MSE = mean_squared_error(df_eval.tax_value, df_eval.y_hat)

    # calculate root mean squared error
    RMSE = math.sqrt(mean_squared_error(df_eval.tax_value, df_eval.y_hat))
    
    # sklearn.metrics.explained_variance_score
    evs = explained_variance_score(df_eval.tax_value, df_eval.y_hat)

    return round(evs,3), (SSE, ESS, TSS, MSE)