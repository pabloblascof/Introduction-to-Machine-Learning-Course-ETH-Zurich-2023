# This serves as a template which will guide you through the implementation of this task. It is advised
# to first read the whole template and get a sense of the overall structure of the code before trying to fill in any of the TODO gaps
# First, we import necessary libraries:
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold 
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge

def fit_2(X_tr, y_tr, lam, X_t):
    """
    This function receives training data points, then fits the ridge regression on this data
    with regularization hyperparameter lambda. The function also receives test data points, and returns 
    the prediction. 

    Parameters
    ----------
    X_tr: matrix of floats, dim = (135,13), inputs with 13 features
    y_tr: array of floats, dim = (135,), input labels)
    lam: float. lambda parameter, used in regularization term
    X_t: matrix of floats, dim = (15,13), inputs with 13 features

    Returns
    ----------
    y_pred: array of floats: dim = (15,), prediction of ridge regression
    """
    # TODO: Enter your code here
    
    # Create Linear Regression Ridge Model
    model = Ridge(alpha = lam, fit_intercept=(False), solver = 'auto', tol = 1e-5, max_iter = 6000)
    
    # Fit moedel to training samples
    model.fit(X_tr,y_tr)

    #Prediction of y
    y_pred = model.predict(X_t)

    assert y_pred.shape == (15,)
    return y_pred


def calculate_RMSE_2(y_true, y_pred):
    """This function calculates RMSE from y_true and y_pred

    Parameters
    ----------
    y_true: array of floats: dim = (15,), true value
    y_pred: array of floats, dim = (15,), prediction

    Returns
    ----------
    RMSE: float: dim = 1, RMSE value
    """
    
    RMSE = 0
    # TODO: Enter your code here
    
    #reshape input
    y_true = y_true.reshape(-1,1)
    y_pred = y_pred.reshape(-1,1)
    
    #calculate root mean squared error (RMSE)
    RMSE = np.sqrt(mean_squared_error(y_true, y_pred))
    
    assert np.isscalar(RMSE)
    return RMSE


def average_LR_RMSE_2(X, y, lambdas, n_folds):
    """
    Main cross-validation loop, implementing 10-fold CV. In every iteration (for every train-test split), the RMSE for every lambda is calculated, 
    and then averaged over iterations.
    
    Parameters
    ---------- 
    X: matrix of floats, dim = (150, 13), inputs with 13 features
    y: array of floats, dim = (150, ), input labels
    lambdas: list of floats, len = 5, values of lambda for which ridge regression is fitted and RMSE estimated
    n_folds: int, number of folds (pieces in which we split the dataset), parameter K in KFold CV
    
    Returns
    ----------
    avg_RMSE: array of floats: dim = (5,), average RMSE value for every lambda
    """
    avg_RMSE = np.zeros(5,)
    RMSE_mat = np.zeros((n_folds, len(lambdas)))
    print(RMSE_mat.shape)

    # TODO: Enter your code here. Hint: Use functions 'fit' and 'calculate_RMSE' with training and test data
    # and fill all entries in the matrix 'RMSE_mat'
    
    #K-Folds cross-validator 
    kf = KFold(n_splits=n_folds, shuffle = True)
    
    #loop over all lambdas and splits
    for k, lam in enumerate(lambdas):
        
        idx_fold = 0
        
        for train_idx, val_idx in kf.split(X):

            # Extract train and validation samples
            X_train = X[train_idx]
            X_val = X[val_idx]
            y_train = y[train_idx]
            y_val = y[val_idx]
            
            #fit data to model
            y_hat = fit_2(X_train, y_train, lam, X_val)
            
            #fit the RMSE between y_val and y_hat    
            RMSE_mat[idx_fold,k] = calculate_RMSE_2(y_val, y_hat)
            idx_fold = idx_fold + 1
        
    #average oder all RSME_mat    
    avg_RMSE = np.mean(RMSE_mat, axis=0)  
    
    assert avg_RMSE.shape == (5,)
    return avg_RMSE


# Main function. You don't have to change this
if __name__ == "__main__":
    # Data loading
    data = pd.read_csv("train.csv")
    y = data["y"].to_numpy()
    #y = y.reshape(-1,1)
    data = data.drop(columns="y")
    
    # print a few data samples
    print(data.head())

    X = data.to_numpy()
    # The function calculating the average RMSE
    lambdas = [0.1, 1, 10, 100, 200]
    n_folds = 10
    avg_RMSE = average_LR_RMSE_2(X, y, lambdas, n_folds)
    # Save results in the required format
    np.savetxt("./results.csv", avg_RMSE, fmt="%.12f")



