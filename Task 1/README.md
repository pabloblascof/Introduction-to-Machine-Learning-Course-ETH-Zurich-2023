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

### Task 1b: Exploring Linear Regression Models

In Task 1b, our objective was to discover a well-fitting linear regression model to achieve a high-performance score. The code implementation involves a "fit" function, which transforms the data into the required format. This transformation is achieved through the "transform_data" function, which reshapes the array into five linear features, five quadratic features, five exponential features, five cosine features, and one constant feature using a loop.

To identify the best linear regression model, similar to Task 1a, we employed the scikit-learn library. We systematically experimented with different models and compared their scores, ultimately selecting the Lasso model as the most suitable option.

These revised descriptions offer a more concise and organized understanding of the tasks and the approach taken.
