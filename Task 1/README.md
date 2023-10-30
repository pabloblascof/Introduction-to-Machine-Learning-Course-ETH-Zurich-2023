### Task 1a: Ridge Regression with Scikit-Learn

Task 1a, was meant to familiarize with the scikit-learn library and perform ridge regression with cross validation. It consists of the following steps:

  1) Import necessary libraries, including Pandas, NumPy, and scikit-learn components for K-Fold cross-validation, mean squared error calculation, and Ridge Regression.

  2) Define functions:
  
    fit_2: Trains a Ridge Regression model with a specified lambda on training data and makes predictions on test data.
    calculate_RMSE_2: Computes the Root Mean Square Error (RMSE) between true and predicted values.
    average_LR_RMSE_2: Performs k-fold cross-validation for Ridge Regression, testing different lambda values and reporting average RMSE values.

  3) Load a dataset from a CSV file, extract the input features and labels.

  4) Call the average_LR_RMSE_2 function with the input data, a list of lambda values, and the number of cross-validation folds.

The script saves the results, i.e., average RMSE values for each lambda, in a CSV file named "results.csv."

### Task 1b: Exploring Feature Transformations

The script preprocesses the input data by transforming it into a more complex feature space, fits a Lasso regression model to this transformed data, and saves the model's parameters (weights) in an output file. This is typically done for the purpose of regression or predictive modeling with potentially more expressive feature representations. It consists of the following steps:
  
  1) Data Transformation: The transform_data function takes a matrix X with 5 input features and transforms these features into a set of 21 new features. These new features include linear, quadratic,   exponential, and cosine transformations of the original features, along with a constant feature. The transformed data is returned as an array of shape (700, 21).
  
  2) Model Fitting: The fit function receives the training data points X and labels y. It first transforms the data using the transform_data function and then fits a Lasso regression model with a       specified alpha (0.3) to the transformed data. It returns the optimal parameters (weights) of the Lasso regression model, which are the coefficients for the 21 transformed features.
  
  3) Data Loading: The script loads a dataset from a CSV file named "train.csv" using Pandas. It extracts the labels (denoted as "y") and the input features while excluding the "Id" column.
  
  The transformed data is used to train a Lasso regression model, and the resulting weights of the model are saved to a file named "results.csv."

