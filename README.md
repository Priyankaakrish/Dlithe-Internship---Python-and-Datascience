DLITHE INTERNSHIP

This internship has trained  different analytics learning library like numpy, pandas, matplotlib and seaborn.
We also had a few classes in machine learning and statistics which help us analyze the given data.

Assignments during Online Internship with DLithe(www.dlithe.com).

Assignment 1

Fish Market 

Database of common fish species for fish market
This dataset is a record of 7 common different fish species in fish market sales. With this dataset, a predictive model can be performed using machine friendly data and estimate the weight of fish can be predicted.

Multiple linear regression is a fundamental practice for this dataset. Multivariate analysis can also be performed.

The dataset is from:https://www.kaggle.com/datasets/aungpyaeap/fish-market?resource=download

1.Google colab was used for Implementation

2.This dataset is a record of 7 common different fish species in fish market sales. With this dataset, a predictive model can be performed using machine friendly data and estimate the weight of fish can be predicted.

2.The libraries used are Pandas for data collection. Seaborn for Visualisation and data analysis. And sklearn for Machine learning algorithm selection.

3.The process starts with gathering the data from kaggle on Fish Market dataset.

4.Cleansing the data checking for any missing values. Removing the unwanted columns from dataset using drop command is part of data processing.

5.Multiple linear regression is a fundamental practice for this dataset. Multivariate analysis can also be performed.


Assignment 2

Vehicle dataset

The data is got from:https://www.kaggle.com/datasets/mayankpatel14/second-hand-used-cars-data-set-linear-regression

1.Spyder was used for Implementation.

2.The libraries used are Pandas for data collection. Seaborn for Visualisation and data analysis. And sklearn for Machine learning algorithm selection.

3.The process starts with gathering the data from kaggle on secondhand car dataset.

4.Cleansing the data checking for any missing values. Removing the unwanted columns from dataset using drop command is part of data processing.

5.Selecting independent variables in x array i.e on road old, on road now, years, km, rating, condition,economy,top speed, hp, torque. And dependent variables or target variable to y array i.e current price. Since the target variable is continuous Linear Regression is planned to be used for prediction.

6.Split universal dataset to trainset and testset using library sklearn,module:model_selection, class : train_test_split. Test size of 0.3 was used with random state of 350. The train set was named x_train, y_train and test set was assigned as x_test, y_test.

7.Linear regression was used as Algorithm Selection for training and predicting the y_test named as y_pred. Using the Library: sklearn, module : Linear_model, class : LinearRegression.

8.Checking the accuracy of y_train andy_pred was done using the Library: sklearn, module : metrics, class : r2score

Assignment 3

GPU Kernel

The dataset https://www.kaggle.com/datasets/rupals/gpu-runtime

*1. *Experiment with various parameters for linear and logistic regression (e.g. learning rate ∝) and report on your findings as how the error/accuracy varies for train and test sets with varying these parameters. Plot the results. Report the best values of the parameters.

2. Experiment with various thresholds for convergence for linear and logistic regression. Plot error results for train and test sets as a function of threshold and describe how varying the threshold affects error. Pick your best threshold and plot train and test error (in one figure) as a function of number of gradient descent iterations.

3. Pick eight features randomly and retrain your models only on these ten features. Compare train and test error results for the case of using your original set of features (14) and eight random features. Report the ten randomly selected features.

4. Now pick eight features that you think are best suited to predict the output, and retrain your models using these ten features. Compare to the case of using your original set of features and to the random features case.

Assignment 4

New york stock Exchange

The dataset https://www.kaggle.com/datasets/dgawlik/nyse

1.The raw, as-is daily prices. Most of data spans from 2010 to the end 2016, for companies new on stock market date range is shorter. There have been approx. 140 stock splits in that time, this set doesn't account for that.

2. The same as prices, but there have been added adjustments for splits.

3.The general description of each company with division on sectors

4.The metrics extracted from annual SEC 10K fillings (2012-2016), should be enough to derive most of popular fundamental indicators.



TECHNICAL INSTRUCTIONS

-> This file contains ipynb file since the code was built on jupyter.

-> Pandas, Numpy, Seaborn and Matplotlib are used for data analysis.

-> Reading the data was done through Pandas.

-> Data Cleaning was done through Pandas and Seaborn.

-> Data testing was done using Numpy.

-> This project was built on python.

-> The dataset is sourced from Kaggle.

-> The results of all questions are based on graphs which are done using matplotlib.




