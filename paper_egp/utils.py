import numpy as np 
from sklearn.utils import check_X_y, check_array
from scipy.spatial.distance import pdist
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr, spearmanr
import scipy as scio
import pandas as pd
import matplotlib.pyplot as plt


def r_assessment(y_pred, y_test, verbose=0):
    y_pred = y_pred.flatten()
    y_test = y_test.flatten()



    df = pd.DataFrame({
        'mae': mean_absolute_error(y_pred, y_test),
        'mse': mean_squared_error(y_pred, y_test),
        'r2': r2_score(y_pred, y_test),
        'rmse': np.sqrt(mean_squared_error(y_pred, y_test))
    }, index=['Results'])


    if verbose:
        print(df.head())

    return df

