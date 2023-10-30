### Task 1a: Linear Regression with Scikit-Learn

To approach Task 1a, we followed the instructions provided and leveraged the scikit-learn library. The main function, "average_LR_RSME_2," conducts K-Folds cross-validation using scikit-learn's KFold function. It iterates through different lambda values and data splits.

Within this loop, two essential functions come into play: "fit_2" and "calculate_RSME_2. "fit_2" is responsible for creating a linear regression model using Ridge regression and fitting it to the training samples. For computing the root mean squared error, "calculate_RSME_2" employs the formula we learned in lectures and task instructions.

The "average_LR_RSME_2" function aggregates the root mean squared errors, and the resulting average is saved to an Excel file named "results."

### Task 1b: Exploring Linear Regression Models

In Task 1b, our objective was to discover a well-fitting linear regression model to achieve a high-performance score. The code implementation involves a "fit" function, which transforms the data into the required format. This transformation is achieved through the "transform_data" function, which reshapes the array into five linear features, five quadratic features, five exponential features, five cosine features, and one constant feature using a loop.

To identify the best linear regression model, similar to Task 1a, we employed the scikit-learn library. We systematically experimented with different models and compared their scores, ultimately selecting the Lasso model as the most suitable option.

These revised descriptions offer a more concise and organized understanding of the tasks and the approach taken.
