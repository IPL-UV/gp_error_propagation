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


def plot_gp(xtest, predictions, std=None, xtrain=None, ytrain=None,  title=None, save_name=None):

    xtest, predictions = xtest.squeeze(), predictions.squeeze()


    fig, ax = plt.subplots()

    # Plot the training data
    if (xtrain is not None) and (ytrain is not None):
        xtrain, ytrain = xtrain.squeeze(), ytrain.squeeze()
        ax.scatter(xtrain, ytrain, s=100, color='r', label='Training Data')

    # plot the testing data
    ax.plot(xtest, predictions, linewidth=5,
            color='k', label='Predictions')

    # plot the confidence interval
    if std is not None:
        std = std.squeeze()
        upper_bound = predictions + 1.960 * std
        lower_bound = predictions - 1.960 * std

        ax.fill_between(xtest, upper_bound, lower_bound,
                        color='red', alpha=0.2, label='95% Condidence Interval')
    # ax.legend()
    if title is not None:
        ax.set_title(title)
        
    ax.tick_params(
    axis='both', 
    which='both',
    bottom=False, 
    top=False, 
    left=False,
    labelleft=False,
    labelbottom=False)

    if save_name:
        fig.savefig(save_name)
    else:
        plt.show()

    return fig