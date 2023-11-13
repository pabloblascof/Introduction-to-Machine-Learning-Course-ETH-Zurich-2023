### Task 2: Data processing and Gaussian Process Regresor

 The core functionalities are as follows:
 
- **Data Loading and Preprocessing:** The script loads training and test data from CSV files. It performs data preprocessing, such as converting categorical data to numerical values and handling missing data using K-nearest neighbors imputation.

- **Model Definition and Training:** The script defines a Gaussian Process Regression (GPR) model with a Matern kernel. It then fits this model to the training data, allowing the model to learn the relationships within the data.

- **Prediction:** Using the trained GPR model, the script makes predictions on the test data, producing an array of predicted values.

- **Results Saving:** : These predictions are saved in a CSV file named "results.csv." This file can be used for further analysis or evaluation of the model's performance.


