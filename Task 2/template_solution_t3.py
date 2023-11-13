# This serves as a template which will guide you through the implementation of this task.  
# First, we import necessary libraries:
import numpy as np
import pandas as pd
import sklearn
from sklearn.gaussian_process import GaussianProcessRegressor
# from sklearn.gaussian_process.kernels import DotProduct, RBF, Polynomial, Matern, RationalQuadratic
from sklearn.gaussian_process.kernels import DotProduct, RBF, Matern, RationalQuadratic
from sklearn.impute import KNNImputer
from sklearn.model_selection import GridSearchCV

# # explicitly require this experimental feature
# from sklearn.experimental import enable_iterative_imputer  # noqa
# # now you can import normally from sklearn.impute
# from sklearn.impute import IterativeImputer

# from fancyimpute import IterativeImputer

# NOTE --> cannot import Polynomial, might need to install.(Carol)

def data_loading():
    """
    This function loads the training and test data, preprocesses it, removes the NaN values and interpolates the missing 
    data using imputation

    Parameters
    ----------
    Returns
    ----------
    X_train: matrix of floats, training input with features
    y_train: array of floats, training output with labels
    X_test: matrix of floats: dim = (100, ?), test input with features
    """
    # Load training data
    train_df = pd.read_csv("train.csv")
    
    print("Training data:")
    print("Shape:", train_df.shape)
    print(train_df.head(2))
    print('\n')
    
    # Load test data
    test_df = pd.read_csv("test.csv")
    
    print("Test data:")
    print(test_df.shape)
    print(test_df.head(2))

    # Dummy initialization of the X_train, X_test and y_train   
    X_train = np.zeros_like(train_df.drop(['price_CHF'],axis=1))
    y_train = np.zeros_like(train_df['price_CHF'])
    X_test = np.zeros_like(test_df)

    # TODO: Perform data preprocessing, imputation and extract X_train, y_train and X_test
    
    # Data preprocessing: turn seasons into numerical values
    train_df.loc[train_df['season'] == 'winter', 'season'] = 0
    train_df.loc[train_df['season'] == 'spring', 'season'] = 1
    train_df.loc[train_df['season'] == 'summer', 'season'] = 2
    train_df.loc[train_df['season'] == 'autumn', 'season'] = 3
    
    test_df.loc[test_df['season'] == 'winter', 'season'] = 0
    test_df.loc[test_df['season'] == 'spring', 'season'] = 1
    test_df.loc[test_df['season'] == 'summer', 'season'] = 2
    test_df.loc[test_df['season'] == 'autumn', 'season'] = 3
    
    # Dropping data
    
    # Dropping the rows with no information at all
    train_df.dropna(how='all', inplace=True)
    test_df.dropna(how='all', inplace=True)
    
    # Dropping the rows with NaN in 'price_CHF' column
    train_df.dropna(subset=['price_CHF'], inplace=True)
    
    # # Drop columns with high nullity
    # thr_p = 0.9     # threshold percentage
    # nan_percentage = train_df.isna().sum() / len(train_df)      # percentage of NaN in each column
    # cols2drop = nan_percentage[nan_percentage > thr_p].index    # columns to drop
    # train_df.drop(columns=cols2drop, inplace=True)              # drop data in training
    # test_df.drop(columns=cols2drop, inplace=True)               # drop data in testing 

    # Filling data 
    
    # # Fill all nan values with mean of the column
    # train_df.fillna(train_df.mean(), inplace=True)
    # test_df.fillna(test_df.mean(), inplace=True)
    
    # Assign train and test data 
    X_train = train_df.drop(['price_CHF'], axis=1)
    y_train = train_df['price_CHF']
    X_test = test_df
    
    #Imputation with KNNImputer
    imputer = KNNImputer(n_neighbors=2)
    X_train = imputer.fit_transform(X_train)
    #y_train = imputer.fit_transform(y_train)
    X_test = imputer.fit_transform(X_test)
    #print("train_df",train_df)
    
    # imputer = IterativeImputer()
    # X_train = imputer.fit(X_train)
    
    # X_train = imputer.transform(X_train)
    # X_test = imputer.transform(X_test)
    
    print(X_train.shape)
    print(y_train.shape)
    print(X_test.shape)
        

    assert (X_train.shape[1] == X_test.shape[1]) and (X_train.shape[0] == y_train.shape[0]) and (X_test.shape[0] == 100), "Invalid data shape"
    return X_train, y_train, X_test

def modeling_and_prediction(X_train, y_train, X_test):
    """
    This function defines the model, fits training data and then does the prediction with the test data 

    Parameters
    ----------
    X_train: matrix of floats, training input with 10 features
    y_train: array of floats, training output
    X_test: matrix of floats: dim = (100, ?), test input with 10 features

    Returns
    ----------
    y_test: array of floats: dim = (100,), predictions on test set
    """

    y_pred=np.zeros(X_test.shape[0])
    
    #TODO: Define the model and fit it using training data. Then, use test data to make predictions
    
    #Matern kernel with hyperparameter fine tuned
    
    kernel = Matern(length_scale=1, nu=2.5)
    
    gpr = GaussianProcessRegressor(kernel, alpha=0.1)

    gpr.fit(X_train, y_train)

    y_pred = gpr.predict(X_test)
    
    assert y_pred.shape == (100,), "Invalid data shape"
    return y_pred

# Main function. You don't have to change this
if __name__ == "__main__":
    # Data loading
    X_train, y_train, X_test = data_loading()
    # The function retrieving optimal LR parameters
    y_pred=modeling_and_prediction(X_train, y_train, X_test)
    # Save results in the required format
    dt = pd.DataFrame(y_pred) 
    dt.columns = ['price_CHF']
    dt.to_csv('results.csv', index=False)
    print("\nResults file successfully generated!")

