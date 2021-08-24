import pandas as pd
import numpy as np

import csv
import os

os.chdir('/Users/Greyson/Desktop/Machine Learning/Final Project/Data')

## load in datasets saved during pre-proessing in R
xtrain = pd.read_csv("xtrain3.csv")
xvalidate = pd.read_csv("xvalidate3.csv")
xtest = pd.read_csv("xtest3.csv")

ytrain = pd.read_csv("ytrain3.csv")
yvalidate = pd.read_csv("yvalidate3.csv")
submission = pd.read_csv('sample_submission.csv')



########## lgb ################

import lightgbm as lgb

# lgb hyper-parameters
params = {'metric': 'rmse',
          'num_leaves': 255,
          'learning_rate': 0.005,
          'feature_fraction': 0.75,
          'bagging_fraction': 0.75,
          'bagging_freq': 5,
          'force_col_wise' : True,
          'random_state': 10}

cat_features = ['shop_id','cityname', 'item_category_id', 'month', 'year','region']


# lgb train and valid dataset
dtrain = lgb.Dataset(xtrain, ytrain)
dvalid = lgb.Dataset(xvalidate, yvalidate)
 
# Train LightGBM model
lgb_model = lgb.train(params=params,
                      train_set=dtrain,
                      num_boost_round=1500,
                      valid_sets=(dtrain, dvalid),
                      early_stopping_rounds=150,
                      categorical_feature=cat_features,
                      verbose_eval=100)      

## make predictions on testing data
preds_lgb = lgb_model.predict(xtest).clip(0,20)

## write to csv
submission['item_cnt_month'] = preds_lgb
submission.to_csv('lgb_submissionv3.csv', index=False)

######### One Hot Encoding ########

## create temporary datasets to merge for coded data set
xtrain_temp = pd.DataFrame(xtrain)
xtrain_temp['what_df'] = "train"
xvalidate_temp = pd.DataFrame(xvalidate)
xvalidate_temp['what_df'] ='validate'
xtest_temp = pd.DataFrame(xtest)
xtest_temp['what_df'] = 'test'


## merge all datasets so when encoded each will have the same levels for all factors
all_df = [xtrain_temp,xvalidate_temp,xtest_temp]
merged= pd.concat(all_df)


## specify features that are categories
cat_features = ['shop_id', 'cityname', 'item_category_id', 'month', 'year','region']

### create dummy variables
all_coded = pd.get_dummies(merged, columns= cat_features)
all_coded = all_coded.reset_index(drop=True)


## split dataframes again and discard row used to determine which dataframe they origniated from
xtrain_coded = all_coded.loc[all_coded['what_df']=='train']
xvalidate_coded = all_coded.loc[all_coded['what_df']=='validate']
xtest_coded = all_coded.loc[all_coded['what_df']=='test']

xtrain_coded = xtrain_coded.drop('what_df',1)
xvalidate_coded = xvalidate_coded.drop('what_df',1)
xtest_coded = xtest_coded.drop('what_df',1)





############ xgb ###########
from xgboost import XGBRegressor


## specify parameters of model
model = XGBRegressor(
    max_depth=7,
    n_estimators=1000,
    min_child_weight=0.5, 
    colsample_bytree=0.8, 
    subsample=0.8, 
    eta=0.1,
    seed=42)

## fit model using encoded train and validation datasets
model.fit(
    xtrain_coded, 
    ytrain, 
    eval_metric="rmse", 
    eval_set=[(xtrain_coded, ytrain), (xvalidate_coded, yvalidate)], 
    verbose=True, 
    early_stopping_rounds = 20)


## make predictions on the testing dataset and write to a csv
Y_pred = model.predict(xvalidate).clip(0, 20)
Y_test = model.predict(xtest).clip(0, 20)

submission = pd.DataFrame({
    "ID": xtest.index, 
    "item_cnt_month": Y_test
})
submission.to_csv('xgb_submission.csv', index=False)





############# random forest ############
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(random_state = 42)

from pprint import pprint

## list all parameters used in a random forest model
print('Parameters currently in use:\n')
pprint(rf.get_params())

from sklearn.model_selection import RandomizedSearchCV
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 5)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree

# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf}
pprint(random_grid)


## create random tree regressor
rf = RandomForestRegressor()


rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 20, cv = 3, verbose=2, random_state=42, n_jobs = -1)

rf_random.fit(xtrain, ytrain)

rf_random.best_params_




#### Fit model
rf = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=7, n_jobs=-1)
rf.fit(xtrain_coded,ytrain.values.ravel())
rf_validate = rf.predict(xvalidate_coded)
from sklearn.metrics import mean_squared_error
from math import sqrt
sqrt(mean_squared_error(rf_validate,yvalidate))


xtest_coded.fillna(0,inplace=True)

predict_test_rf = rf.predict(xtest_coded)

np.any(np.isnan(xtest_coded))
np.all(np.isfinite(xtest_coded))
xtest_coded.dtypes

submission_rf = pd.DataFrame({
    "ID": xtest.index, 
    "item_cnt_month": predict_test_rf
})
submission_rf.to_csv('rf_submissionv3.csv', index=False)
