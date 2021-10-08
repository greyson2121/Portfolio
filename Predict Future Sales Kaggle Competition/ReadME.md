This analysis is from participating in a Kaggle competition.  It is focused on predicting the number of sales of items in Russian shops in a calendar month.  The training data set consists of 33 months of data from February 2013 to October 2015.  There are 60 shops in total selling from a selection of thousands of items.  Using this data, the goal is to predict the total sales of an item in a store during the month of November 2015.  The only predictor variables given to participants were item id, shop id, item category, data, and store name which means there is a signficant amount of feature engineering necessary to create a model that performs well.  The data is orignially split into six different files and reflects daily sales of items, so data cleaning and aggregation to monthly sales is necessary before any analysis can be performed.  After pre-processing a variety of models were fit to the data including lasso regressions, ridge regression, xgboost, light gbm, and random forest.

The pre processing of the data was done in R and can be found at:  https://rpubs.com/greyson21/801512

Due to the size of the data set, using R to build the models for analysis became untenable and inefficient.  This process was eventually moved to python.

The Kaggle competition and all data can be found at: https://www.kaggle.com/c/competitive-data-science-predict-future-sales/overview
