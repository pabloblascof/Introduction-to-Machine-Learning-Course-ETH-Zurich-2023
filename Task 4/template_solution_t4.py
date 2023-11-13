# This serves as a template which will guide you through the implementation of this task.
# First, we import necessary libraries:
import pandas as pd
import numpy as np
import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler 

from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.utils as utils

from sklearn.model_selection import GridSearchCV


import os

# Set the device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load_data():
    """
    This function loads the data from the csv files and returns it as numpy arrays.

    input: None
    
    output: x_pretrain: np.ndarray, the features of the pretraining set
            y_pretrain: np.ndarray, the labels of the pretraining set
            x_train: np.ndarray, the features of the training set
            y_train: np.ndarray, the labels of the training set
            x_test: np.ndarray, the features of the test set
    """
    x_pretrain = pd.read_csv("public/pretrain_features.csv.zip", index_col="Id", compression='zip').drop("smiles", axis=1).to_numpy()
    y_pretrain = pd.read_csv("public/pretrain_labels.csv.zip", index_col="Id", compression='zip').to_numpy().squeeze(-1)
    x_train = pd.read_csv("public/train_features.csv.zip", index_col="Id", compression='zip').drop("smiles", axis=1).to_numpy()
    y_train = pd.read_csv("public/train_labels.csv.zip", index_col="Id", compression='zip').to_numpy().squeeze(-1)
    x_test = pd.read_csv("public/test_features.csv.zip", index_col="Id", compression='zip').drop("smiles", axis=1)
    return x_pretrain, y_pretrain, x_train, y_train, x_test

class Net(nn.Module):
    """
    The model class, which defines our feature extractor used in pretraining.
    """
       
    def __init__(self):
        """
        The constructor of the model.
        """
        # super().__init__()
        # TODO: Define the architecture of the model. It should be able to be trained on pretraing data 
        # and then used to extract features from the training and test data.
        
        super(Net, self).__init__()
        self.fc1 = nn.Linear(1000, 6000)
        self.bn1 = nn.BatchNorm1d(6000)
        self.fc2 = nn.Linear(6000, 1000)
        self.bn2 = nn.BatchNorm1d(1000)
        self.fc3 = nn.Linear(1000, 500)
        self.bn3 = nn.BatchNorm1d(500)
        self.dropout = nn.Dropout(p=0.5)
        self.out = nn.Linear(500, 1)

    def forward(self, x, include_last=True):
        x = self.bn1(self.fc1(x))
        x = F.relu(x)
        x = self.dropout(x)
        x = self.bn2(self.fc2(x))
        x = F.relu(x)
        x = self.dropout(x)
        x = self.bn3(self.fc3(x))

        if include_last:
            x = F.relu(x)
            x = self.dropout(x)
            x = self.out(x)
        
        return x

    
def make_feature_extractor(x, y, batch_size=512, eval_size=1000):
    """
    This function trains the feature extractor on the pretraining data and returns a function which
    can be used to extract features from the training and test data.

    input: x: np.ndarray, the features of the pretraining set
              y: np.ndarray, the labels of the pretraining set
                batch_size: int, the batch size used for training
                eval_size: int, the size of the validation set
            
    output: make_features: function, a function which can be used to extract features from the training and test data
    """
    # Pretraining data loading
    in_features = x.shape[-1]
    x_tr, x_val, y_tr, y_val = train_test_split(x, y, test_size=eval_size, random_state=0, shuffle=True)
    x_tr, x_val = torch.tensor(x_tr, dtype=torch.float).to(device), torch.tensor(x_val, dtype=torch.float).to(device)
    y_tr, y_val = torch.tensor(y_tr, dtype=torch.float).to(device), torch.tensor(y_val, dtype=torch.float).to(device)
    
    # Create DataLoaders for training
    dataset = TensorDataset(x_tr,y_tr)
    train_loader = DataLoader(dataset=dataset,
                        batch_size=batch_size,
                        shuffle = True)
    
    # Create DataLoaders for validation
    dataset = TensorDataset(x_val,y_val)
    val_loader = DataLoader(dataset=dataset,
                        batch_size=batch_size,
                        shuffle = True)
    
    # Model declaration
    model = Net()
    model.train()
    model.to(device)
    
    # Loss function and optimizer
    learning_rate = 0.001
    weight_decay = 0.001
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # TODO: Implement the training loop. The model should be trained on the pretraining data. Use validation set 
    # to monitor the loss.
    
    # Load the value of previous max_val_loss from the file if it exists
    ref_val_loss_file = "ref_val_loss.txt"
    if os.path.exists(ref_val_loss_file):
        with open(ref_val_loss_file, 'r') as f:
            ref_val_loss = float(f.read())
    else:
        ref_val_loss = float(1000000)
        
    print(ref_val_loss)

    num_epochs = 15
    
    for epoch in range(num_epochs):
        
        print(epoch)
        train_loss = 0
        val_loss = 0
        
        # Training
        
        for i, [X,y] in enumerate(train_loader):
            
            optimizer.zero_grad()
            
            # Forward pass
            y_pred = model(X)
            y_pred = y_pred.squeeze(1)
            loss = criterion(y_pred, y)
            train_loss += loss.item()
            
            # Backward and optimize
            loss.backward()
            optimizer.step()
            
        print(f"Training loss : {train_loss / (i+1)}")
        
        # Validation
        
        model.eval()
        
        with torch.no_grad():
            for i, [X,y] in enumerate(val_loader):
                
                # Forward pass
                y_pred = model.forward(X)
                y_pred = y_pred.squeeze(1)
                loss = criterion(y_pred, y)
                val_loss += loss.item()

        print(f"Validation loss : {val_loss / (i+1)}")
        
        avg_val_loss = val_loss / (i+1)
        
        # Save best result parameters and loss
        
        if avg_val_loss < ref_val_loss:
            print(f"New best val_loss : {avg_val_loss}")
            ref_val_loss = avg_val_loss
            torch.save(model.state_dict(), "best_params.pt")
            with open(ref_val_loss_file, 'w') as f:
                f.write(str(ref_val_loss))

    if os.path.exists("best_params.pt"):
      print('Best P Train')
  
    model = Net()
    model.to(device)
    model.load_state_dict(torch.load("best_params.pt"))

    def make_features(x):
        """
        This function extracts features from the training and test data, used in the actual pipeline 
        after the pretraining.

        input: x: np.ndarray, the features of the training or test set

        output: features: np.ndarray, the features extracted from the training or test set, propagated
        further in the pipeline
        """
        # Instantiate model and load with best parameters
        model = Net()
        model.to(device)
        model.load_state_dict(torch.load("best_params.pt"))   
        model.eval()
        if os.path.exists("best_params.pt"):
          print('Best P')
        
        if isinstance(x, pd.DataFrame):
            x = x.values
        
        # Compute the features (forward pass through network except for last layer)
        with torch.no_grad():
            x_torch = torch.tensor(x, dtype=torch.float).to(device)
            x_ftr = model.forward(x_torch, include_last=False).to('cpu')

        
        return x_ftr.detach().numpy()

    return make_features

def make_pretraining_class(feature_extractors):
    """
    The wrapper function which makes pretraining API compatible with sklearn pipeline
    
    input: feature_extractors: dict, a dictionary of feature extractors

    output: PretrainedFeatures: class, a class which implements sklearn API
    """

    class PretrainedFeatures(BaseEstimator, TransformerMixin):
        """
        The wrapper class for Pretraining pipeline.
        """
        def __init__(self, *, feature_extractor=None, mode=None):
            self.feature_extractor = feature_extractor
            self.mode = mode

        def fit(self, X=None, y=None):
            return self

        def transform(self, X):
            assert self.feature_extractor is not None
            # Access the feature extractor function correctly using the feature_extractor name
            X_new = feature_extractors[self.feature_extractor](X)
            return X_new
        
    return PretrainedFeatures


def get_regression_model():
    """
    This function returns the regression model used in the pipeline.

    input: None

    output: model: sklearn compatible model, the regression model
    """
    # TODO: Implement the regression model. It should be able to be trained on the features extracted
    # by the feature extractor.
    
    # Ridge regression model
    model = Ridge(fit_intercept=(True))
    
    return model

# Main function. You don't have to change this
if __name__ == '__main__':
    # Load data
    x_pretrain, y_pretrain, x_train, y_train, x_test = load_data()
    print("Data loaded!")
    
    # Utilize pretraining data by creating feature extractor which extracts lumo energy 
    # features from available initial features
    feature_extractor =  make_feature_extractor(x_pretrain, y_pretrain)
    PretrainedFeatureClass = make_pretraining_class({"pretrain": feature_extractor})
    
    # regression model
    regression_model = get_regression_model()

    y_pred = np.zeros(x_test.shape[0])
    
    # TODO: Implement the pipeline. It should contain feature extraction and regression. You can optionally
    # use other sklearn tools, such as StandardScaler, FunctionTransformer, etc.
    
    # Feature extraction
    x_train_features = feature_extractor(x_train)
    x_test_features = feature_extractor(x_test)
    
    # Standardize the features
    scaler = StandardScaler()
    x_train_features_scaled = scaler.fit_transform(x_train_features)
    x_test_features_scaled = scaler.transform(x_test_features)
    
    # Regression - Perform Grid Search for best hyperparameters
    param_grid = {
    'alpha': [0.1, 1, 10, 100, 1000],
    'fit_intercept': [True, False],
    'solver': ['auto'],
    'tol': [1e-4, 1e-5, 1e-6],
    'max_iter': [1000, 5000, 10000]
}
    regression_model = get_regression_model()
    grid_search = GridSearchCV(regression_model, param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(x_train_features_scaled, y_train)
    
    # Print best hyperparameters
    print("Best Hyperparameters:")
    print(grid_search.best_params_)
    
    # Regression with best hyperparameters
    best_regression_model = grid_search.best_estimator_
    
    best_regression_model.fit(x_train_features_scaled, y_train)
    y_pred = best_regression_model.predict(x_test_features_scaled)

    assert y_pred.shape == (x_test.shape[0],)
    y_pred = pd.DataFrame({"y": y_pred}, index=x_test.index)
    y_pred.to_csv("results.csv", index_label="Id")
    print("Predictions saved, all done!")

    
    