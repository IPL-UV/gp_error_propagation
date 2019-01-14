import warnings
warnings.simplefilter('ignore', category=[DeprecationWarning, FutureWarning])


import numpy as np
import time
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C 
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from paper_egp.utils import plot_gp, r_assessment
from paper_egp.egp import NIGP
from scipy import interpolate
import matplotlib.pyplot as plt

try:
    from gp_extras import HeteroscedasticKernel
    from sklearn.cluster import KMeans
    extras_install = True
except ImportError:
    print("GP Extras file not found. Won't do Example")
    extras_install = False






class Example1D(object):
    def __init__(self, func=1, x_cov=0.3):
        self.func = func
        self.x_cov = x_cov
        self.models = None
        self.data = None
        self.n_restarts = 10
        self.models_fitted = True
        self.empirical_variance_fitted = True
        self.average_scores_fitted = None
        self.fig_save_1d = "/figures/experiment_1d"
        self.fig_emp_error = "/home/emmanuel/projects/error_propagation/figures/paper/experiment_1d/"

    def get_data(self, func=None, x_error=None):
        
        if func is None:
            func = self.func
        if x_error is None:
            x_error = self.x_cov

        X, y, error_params = example_error_1d(func, x_error)

        self.X = X
        self.y = y
        self.error_params = error_params
        self.x_cov = error_params['x']
        self.sigma_y = error_params['y']
        self.f = error_params['f']
        
        self.data = True
        
        return self

    def get_gp_models(self):

        if self.data is not True:
            self.get_data()

        self.models = get_models(self.X['train'], self.y['train'], x_cov=self.x_cov)

        return self

    def fit_gps(self):

        if self.models is None:
            self.get_gp_models()


        df = pd.DataFrame(columns=['model', 'mae', 'mse', 'rmse', 'r2'])

        for imodel in self.models.keys():
            
            # Make Predictions
            y_pred  = self.models[imodel].predict(self.X['test'])

            # Get Error Stats
            mae, mse, rmse, r2 = r_assessment(y_pred, self.y['test'], verbose=0)

            df = df.append({
                'model': imodel,
                'mae': mae,
                'mse': mse,
                'rmse': rmse,
                'r2': r2
            }, ignore_index=True)

        self.results = df
        self.models = self.models

        self.models_fitted = True

        return self

    def show_gp_fit(self, show=True):

        if self.models_fitted is not True:
            self.fit_gps()


        for imodel in self.models.keys():            # Plot

            # Get plot data
            mean, std = self.models[imodel].predict(self.X['plot'], return_std=True)

            save_name = self.fig_save_1d + 'gp_' + imodel + '.png'
            plot_gp(self.X['plot'], mean, 
                    std=std, xtrain=self.X['train'], 
                    ytrain=self.y['train'],  
                    save_name=save_name)


        return self

    def get_empirical_variance(self, n_points=1000, n_trials=100):

        if self.models_fitted is not True:
            self.fit_gps()
        
        rng = np.random.RandomState(None)


        #
        mae_score = {ikey: list() for ikey in self.models.keys()}
        mse_score = {ikey: list() for ikey in self.models.keys()}
        abs_error = {ikey: list() for ikey in self.models.keys()}
        squared_error = {ikey: list() for ikey in self.models.keys()}

        x = np.linspace(self.X['plot'].min(), self.X['plot'].max(), n_points)

        # Testing set (noise-less)
        ytest = self.f(x)
        ytest += self.sigma_y * rng.randn(n_points)
        ytest = ytest.reshape(-1, 1)

        # loop through trials
        for itrial in range(n_trials):
            if itrial % 10 == 0:
                print('Trial: {}'.format(itrial + 1))

            # Generate x samples with random error
            xtest = x + self.x_cov * rng.randn(n_points)
            xtest = xtest.reshape(-1, 1)

            # Loop through model
            for imodel in self.models.keys():

                mean = self.models[imodel].predict(xtest)

                abs_error[imodel].append(np.abs(mean.squeeze() - ytest.squeeze()))
                squared_error[imodel].append((mean.squeeze() - ytest.squeeze())**2)
                mae_score[imodel].append(
                    mean_absolute_error(mean.squeeze(), ytest.squeeze()))
                mse_score[imodel].append(
                    mean_squared_error(mean.squeeze(), ytest.squeeze()))
        # Convert to arrays

        for imodel in self.models.keys():
            abs_error[imodel] = np.array(abs_error[imodel])
            squared_error[imodel] = np.array(squared_error[imodel])
            mae_score[imodel] = np.array(mae_score[imodel])
            mse_score[imodel] = np.array(mse_score[imodel])  

             
        self.abs_error = abs_error
        self.squared_error = squared_error
        self.mae_score = mae_score
        self.mse_score = mse_score

        self.empirical_variance_fitted = True
        return self

    def get_average_empirical(self):
        if self.empirical_variance_fitted is not True:
            self.get_empirical_variance()

        avg_abs_error = dict()
        avg_squared_error = dict()
        avg_mae_score = dict()
        avg_mse_score = dict()

        for imodel in self.models.keys():

            avg_abs_error[imodel] = np.mean(
                np.array(self.abs_error[imodel]).squeeze(), axis=0)
            avg_squared_error[imodel] = np.mean(
                np.array(self.squared_error[imodel]).squeeze(), axis=0)
            avg_mae_score[imodel] = np.mean(np.array(self.mae_score[imodel]))
            avg_mse_score[imodel] = np.mean(np.array(self.mse_score[imodel]))

        self.avg_abs_error = avg_abs_error
        self.avg_squared_error = avg_squared_error
        self.avg_mae_score = avg_mae_score
        self.avg_mse_score = avg_mse_score

        self.average_scores_fitted = True

        return self

    def average_empirical_errors(self, metric='mse', with_sigma=False):
        
        if self.average_scores_fitted is not True:
            self.get_average_empirical()

        for imodel in self.models.keys():
            self.plot_average_empirical(imodel, metric=metric, with_sigma=with_sigma)

        return None

    def empirical_errors(self):

        for imodel in self.models.keys():

            self.plot_empirical(imodel)
        

        return self

    def plot_empirical(self, model_name, show=True):


        x = np.linspace(self.X['plot'].min(), self.X['plot'].max(), 1000)

        fig, ax = plt.subplots()

        pred, std = self.models[model_name].predict(x[:, np.newaxis], return_std=True)

        for sq_err in self.squared_error[model_name]:
            ax.scatter(x, sq_err, s=0.05, color='k')

        ax.plot(x, std**2, linewidth=4, color='r', label='Predictive Variance')
        ax.legend(fontsize=20)
        ax.grid(True)
        ax.tick_params(
            axis='both', 
            which='both',
            bottom=False, 
            top=False, 
            left=False,
            labelleft=False,
            labelbottom=False)

        save_name = self.fig_emp_error + model_name + '_emp_variance.png'           
        fig.savefig(save_name, bbox_inhces='tight',
                    dpi=100, frameon=None)

        if show:
            plt.show()
        else:
            plt.close()


        return None

    def plot_average_empirical(self, model_name, show=True, metric='mse', with_sigma=False):

        x = np.linspace(-10, 10, 1000)
        pred, std = self.models[model_name].predict(x[:, np.newaxis], return_std=True)
        if metric=='mse':
            error_line = interpolate.interp1d(
                x, self.avg_squared_error[model_name], kind='slinear')(x)
        else:
            error_line = interpolate.interp1d(
                x, self.avg_abs_error[model_name], kind='slinear')(x)

        fig, ax = plt.subplots()

        if not with_sigma:
            sigma_y = self.models['mean'].kernel_.get_params()['k2__noise_level']
        else:
            sigma_y = 0.0
        if metric=='mse':
            ax.plot(x, error_line, linewidth=2,
                    color='k', label='Average Squared Error')
            ax.plot(x, std**2 - sigma_y, linewidth=4, color='r', label='Predictive Variance')
            # ax.legend(['Mean Squared Error', 'Predictive Variance'], fontsize=12)
        else:
            ax.plot(x, error_line, linewidth=2,
                    color='k', label='Average Absolute Error')
            ax.plot(x, (std - np.sqrt(sigma_y)), linewidth=4, color='r', label='Predictive Standard Deviation')
            # ax.legend(['Mean Absolute Error', 'Predictive Standard Deviation'], fontsize=12)            
        ax.grid(True)
        ax.tick_params(
            axis='both', 
            which='both',
            bottom=False, 
            top=False, 
            left=False,
            labelleft=False,
            labelbottom=False)
        plt.show()
        if metric=='mse':
            save_name = (f'{model_name}_{metric}_avg_emp_variance.png')
        else:
            save_name = (f'{model_name}_{metric}_avg_emp_std.png')        
        fig.savefig(self.fig_emp_error + save_name, bbox_inhces='tight',
                    dpi=100, frameon=None)
        if show:
            plt.show()
        else:
            plt.close()

        return None





def get_models( xtrain, ytrain, x_cov=None):

    gp_models = dict()

    # ================
    # Standard GP
    # ================
    print('Fitting standard GP...')
    kernel = C() * RBF() + WhiteKernel()
    simple = GaussianProcessRegressor(kernel=kernel,
                                            normalize_y=True,
                                            n_restarts_optimizer=10,
                                            random_state=123)
    simple.fit(xtrain, ytrain)
    gp_models['standard'] = simple


    # ==============================
    # My Simple Predictive Variance
    # ==============================
    print('Fitting my GP with the Predictive Variance...')
    mean = NIGP(kernel=gp_models['standard'].kernel_,
                         x_cov=x_cov, 
                         n_restarts_optimizer=0,
                         normalize_y=True,
                         var_method = 'mean',
                         random_state=123)
    mean.fit(xtrain, ytrain)
    gp_models['mean'] = mean

    # =======================
    # Heteroscedastic Noise
    # =======================

    if extras_install:
        print('Fitting GP with Heteroscedastic Kernel...')
        prototypes = KMeans(n_clusters=5).fit(xtrain).cluster_centers_
        kernel = C() * RBF() + HeteroscedasticKernel.construct(prototypes)
        hetero = GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=10)
        hetero.fit(xtrain, ytrain)
        gp_models['hetero'] = hetero

    return gp_models

def empirical_variance_exp(models, X, error_params, x_error=None, n_points=1000, n_trials=100):

    rng = np.random.RandomState(None)

    f = error_params['f']
    sigma_y = error_params['y']

    if x_error is None:
        sigma_x = error_params['x']
    else:
        sigma_x = x_error

    #
    mae_score = {ikey: list() for ikey in models.keys()}
    mse_score = {ikey: list() for ikey in models.keys()}
    abs_error = {ikey: list() for ikey in models.keys()}
    squared_error = {ikey: list() for ikey in models.keys()}

    x = np.linspace(X.min(), X.max(), n_points)
    
    # Testing set (noise-less)
    ytest = f(x)
    ytest += sigma_y * rng.randn(n_points)
    ytest = ytest.reshape(-1, 1)
    
    # loop through trials
    for itrial in range(n_trials):
        if itrial % 10 == 0:
            print('Trial: {}'.format(itrial + 1))

        # Generate x samples with random error
        xtest = x + sigma_x * rng.randn(n_points)
        xtest = xtest.reshape(-1, 1)

        # Loop through model
        for imodel in models.keys():

            mean = models[imodel].predict(xtest)

            abs_error[imodel].append(np.abs(mean.squeeze() - ytest.squeeze()))
            squared_error[imodel].append((mean.squeeze() - ytest.squeeze())**2)
            mae_score[imodel].append(
                mean_absolute_error(mean.squeeze(), ytest.squeeze()))
            mse_score[imodel].append(
                mean_squared_error(mean.squeeze(), ytest.squeeze()))
    # Convert to arrays
    # Convert to arrays

    for imodel in models.keys():
        abs_error[imodel] = np.array(abs_error[imodel])
        squared_error[imodel] = np.array(squared_error[imodel])
        mae_score[imodel] = np.array(mae_score[imodel])
        mse_score[imodel] = np.array(mse_score[imodel])
    
    return abs_error, squared_error, mae_score, mse_score

def run_fit_exp(gp_models, X, y, error_params):

    fig_save = "/home/emmanuel/projects/2018_igarss/figures/error/1d_example/"

    df = pd.DataFrame(columns=['model', 'mae', 'mse', 'rmse', 'r2'])

    for imodel in gp_models.keys():
        
        # Make Predictions
        y_pred  = gp_models[imodel].predict(X['test'])

        # Get Error Stats
        mae, mse, rmse, r2 = r_assessment(y_pred, y['test'], verbose=0)

        # Get plot data
        mean, std = gp_models[imodel].predict(X['plot'], return_std=True)

        # Plot
        save_name = fig_save + 'gp_' + imodel + '.png'
        plot_gp(X['plot'], mean, std=std, xtrain=X['train'], ytrain=y['train'],  save_name=save_name)

        df = df.append({
            'model': imodel,
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'r2': r2
        }, ignore_index=True)

    df_path = '/home/emmanuel/projects/2018_igarss/data/results/1d_example/1d_gp.pckl'
    df.to_pickle(df_path)

    return gp_models

def plot_empirical_error(model, X, squared_error, save_name=None, show=True):

    fig_save = "/home/emmanuel/projects/2018_igarss/figures/error/1d_example/"

    x = np.linspace(X.min(), X.max(), 1000)

    fig, ax = plt.subplots()

    pred, std = model.predict(x[:, np.newaxis], return_std=True)

    for sq_err in squared_error:
        ax.scatter(x, sq_err, s=0.05, color='k')

    ax.plot(x, std**2, linewidth=4, color='r', label='Predictive Variance')
    ax.legend(fontsize=20)
    ax.grid(True)


    if save_name is not None:
        fig.savefig(fig_save + save_name + '.png', bbox_inhces='tight',
                    dpi=100, frameon=None)

    if show:
        plt.show()


    return None

def get_average_error(abs_error, squared_error, mae_score, mse_score):

    avg_abs_error = dict()
    avg_squared_error = dict()
    avg_mae_score = dict()
    avg_mse_score = dict()

    for imodel in abs_error.keys():

        avg_abs_error[imodel] = np.mean(
            np.array(abs_error[imodel]).squeeze(), axis=0)
        avg_squared_error[imodel] = np.mean(
            np.array(squared_error[imodel]).squeeze(), axis=0)
        avg_mae_score[imodel] = np.mean(np.array(mae_score[imodel]))
        avg_mse_score[imodel] = np.mean(np.array(mse_score[imodel]))

    return avg_abs_error, avg_squared_error, avg_mae_score, avg_mse_score


def example_error_1d(func=1, x_error=0.3):
    seed = 123
    rng = np.random.RandomState(seed=seed)

    # sample data parameters
    n_train, n_test, n_trial = 60, 100, 2000
    sigma_y = 0.05
    x_cov = x_error
    x_min, x_max = -10, 10

    # real function
    if func == 1:
        f = lambda x: np.sin(1.0 * np.pi / 1.6 * np.cos(5 + .5 * x))
    elif func == 2:
        f = lambda x: np.sinc(x)

    else:
        f = lambda x: np.sin(2. * x) + np.exp(0.2 * x)

    # Training add x, y = f(x)
    x = np.linspace(x_min, x_max, n_train + n_test)

    print(n_train)
    x, xs, = train_test_split(x, train_size=n_train, random_state=seed)

    # add noise
    y = f(x)
    x_train = x + x_cov * rng.randn(n_train)
    y_train = f(x) + sigma_y * rng.randn(n_train)

    x_train, y_train = x_train[:, np.newaxis], y_train[:, np.newaxis]

    # -----------------
    # Testing Data
    # -----------------

    ys = f(xs)

    # Add noise
    x_test = xs + x_cov * rng.randn(n_test)
    y_test = ys

    x_test, y_test = x_test[:, np.newaxis], y_test[:, np.newaxis]

    # -------------------
    # Plot Points
    # -------------------
    x_plot = np.linspace(x_min, x_max, n_test)[:, None]
    y_plot = f(x_plot)

    X = {
        'train': x_train,
        'test': x_test,
        'plot': x_plot
    }
    y = {
        'train': y_train,
        'test': y_test,
        'plot': y_plot
    }

    error_params = {
        'x': x_cov,
        'y': sigma_y,
        'f': f
    }

    return X, y, error_params




def main():

    from paper_egp.experiment_1d import Example1D as Experiment1D
    error_exp = Experiment1D(func=1, x_cov=0.3)

    error_exp.fit_gps()


    pass




if __name__ == '__main__':
    main()