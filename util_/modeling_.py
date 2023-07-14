# data manipulation
import pandas as pd
import numpy as np

# data visualization
import matplotlib.pyplot as plt
import seaborn as sns

# modeling
from sklearn.linear_model import LinearRegression, LassoLars, TweedieRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import explained_variance_score

# statistics testing
import scipy.stats as stats

# system manipulation
import itertools
import os
import sys
sys.path.append("./util_")
import prepare_
import explore_

# other
import env
import warnings
warnings.filterwarnings("ignore")

# set the random seed
np.random.seed(95)

##################################
# This data is already been split and save
# This is only training data
train_scaled = pd.read_csv("./00_project_data/1-1_training_data.csv", index_col=0)
validate_scaled = pd.read_csv("./00_project_data/1-2_validation_data.csv", index_col=0)

train_scaled = train_scaled.reset_index(drop=True)
validate_scaled = validate_scaled.reset_index(drop=True)
train_scaled.head()

# separate features from target
# these coluns are set in order of importance
xtrain = train_scaled[['los_angeles', 'ventura', "orange",
                       'bathrooms_scaled','sqr_feet_scaled', "bedrooms_scaled", "year_built_scaled"]]

ytrain= train_scaled.tax_value

### ----------------------------------------------------------------

# separate features from target
# these coluns are set in order of importance
xval = validate_scaled[['los_angeles', 'ventura', "orange",
                       'bathrooms_scaled','sqr_feet_scaled', "bedrooms_scaled", "year_built_scaled"]]

yval= validate_scaled.tax_value



###################################
# This is only the test data
test_scaled = pd.read_csv("./00_project_data/1-3_testing_data.csv", index_col=0)

test_scaled = test_scaled.reset_index(drop=True)

# separate features from target
# these coluns are set in order of importance
xtest = test_scaled[['los_angeles', 'ventura', 'orange', 
                       'bathrooms_scaled','sqr_feet_scaled', 
                       'bedrooms_scaled', 'year_built_scaled']]

ytest= test_scaled.tax_value
##################################


# create a temporary dataframe for the baseline
base = pd.DataFrame({
    "mean_baseline": np.arange(len(train_scaled)),
    "median_baseline": np.arange(len(train_scaled))
})

# get the baseline averages
mean_base = train_scaled.tax_value.mean()
median_base = train_scaled.tax_value.median()

# add the averages into the dataframe
base.mean_baseline = mean_base
base.median_baseline = median_base

# compute the RMSE baseline
# set baseline at the mean and median of the target
baseline_mean = mean_squared_error(ytrain,base.mean_baseline) ** (0.5)
baseline_median = mean_squared_error(ytrain, base.median_baseline) ** (0.5)


def get_ols():
    """
    return the linear model results
    """
    # MAKE THE THING: create the model object
    linear_model_ols = LinearRegression(fit_intercept=True)

    #1. FIT THE THING: fit the model to training data
    OLSmodel = linear_model_ols.fit(xtrain, ytrain)

    #2. USE THE THING: make a prediction
    ytrain_pred_ols = linear_model_ols.predict(xtrain)

    #3. Evaluate: RMSE
    rmse_train_ols = mean_squared_error(ytrain, ytrain_pred_ols) ** (.5) # 0.5 to get the root


    #2. USE THE THING: make a prediction
    yval_pred_ols = linear_model_ols.predict(xval)

    #3. Evaluate: RMSE
    rmse_val_ols = mean_squared_error(yval, yval_pred_ols) ** (.5) # 0.5 to get the root


    # Create a DataFrame with a single row
    data = {'RMSE Training': rmse_train_ols,
            'RMSE Validation': rmse_val_ols, 
            'train validate RMSE diff': rmse_val_ols - rmse_train_ols,
        'val baseline diff': rmse_val_ols - baseline_mean,
        'R2_validate': explained_variance_score(yval, yval_pred_ols)}
    results = pd.DataFrame([data])
    return results

def get_lossa_lars():
    """
    return the lossa + lars predictions
    """
    # MAKE THE THING: create the model object
    linear_nodel_lars = LassoLars(alpha= 1.0, fit_intercept=False, max_iter=1000)

    #1. FIT THE THING: fit the model to training data
    # We must specify the column in y_train, since we have converted it to a dataframe from a series!
    laslars = linear_nodel_lars.fit(xtrain, ytrain)

    #2. USE THE THING: make a prediction
    ytrain_pred_lars = linear_nodel_lars.predict(xtrain)

    #3. Evaluate: RMSE
    rmse_train_lars = mean_squared_error(ytrain, ytrain_pred_lars) ** (0.5)

    # predict validate
    yval_pred_lars = linear_nodel_lars.predict(xval)

    # evaluate: RMSE
    rmse_val_lars = mean_squared_error(yval, yval_pred_lars) ** (0.5)

    # how important is each feature to the target
    laslars.coef_

    # Create a DataFrame with a single row
    data = {'RMSE Training': rmse_train_lars,
            'RMSE Validation': rmse_val_lars, 
            'RMSE baseline': baseline_mean,
            'train validate RMSE diff': rmse_val_lars - rmse_train_lars,
        'val baseline diff': rmse_val_lars - baseline_mean,
        'R2_validate': explained_variance_score(yval, yval_pred_lars)}
    results = pd.DataFrame([data])
    return results

def get_tweedie():
    """
    return the tweedie predictions
    """
    # MAKE THE THING: create the model object
    linear_nodel_twd = TweedieRegressor(alpha= 1.0, power= 1)

    #1. FIT THE THING: fit the model to training data
    # We must specify the column in y_train, since we have converted it to a dataframe from a series!
    tweedieReg = linear_nodel_twd.fit(xtrain, ytrain)

    #2. USE THE THING: make a prediction
    ytrain_pred_twd = linear_nodel_twd.predict(xtrain)

    #3. Evaluate: RMSE
    rmse_train_twd = mean_squared_error(ytrain, ytrain_pred_twd) ** (0.5)

    # predict validate
    yval_pred_twd = linear_nodel_twd.predict(xval)

    # evaluate: RMSE
    rmse_val_twd = mean_squared_error(yval, yval_pred_twd) ** (0.5)

    # how important is each feature to the target
    tweedieReg.coef_


    # Create a DataFrame with a single row
    data = {'RMSE Training': rmse_train_twd,
            'RMSE Validation': rmse_val_twd,
            'RMSE baseline': baseline_mean,
            'train validate RMSE diff': rmse_val_twd - rmse_train_twd,
        'val baseline diff': rmse_val_twd - baseline_mean,
        'R2_validate': explained_variance_score(yval, yval_pred_twd)}
    results = pd.DataFrame([data])
    return results


def get_poly_predictions():
    """
    return the plynomial regression predictions
    """
    #1. Create the polynomial features to get a new set of features
    pf = PolynomialFeatures(degree=3) #Quadratic aka x-squared

    #1. Fit and transform X_train_scaled
    xtrain_degree2 = pf.fit_transform(xtrain)

    #1. Transform X_validate_scaled & X_test_scaled 
    xval_degree2 = pf.transform(xval)

    xtrain_degree2[1]
    # x_test_degree2 = pf.transform(x_test_scaled)

    #2.1 MAKE THE THING: create the model object
    linear_model_pf = LinearRegression()

    #2.2 FIT THE THING: fit the model to our training data. We must specify the column in y_train, 
    # since we have converted it to a dataframe from a series! 
    polyFeat = linear_model_pf.fit(xtrain_degree2, ytrain)

    #3. USE THE THING: predict train
    ytrain_pred_poly = linear_model_pf.predict(xtrain_degree2)

    #4. Evaluate: rmse
    rmse_train_poly = mean_squared_error(ytrain, ytrain_pred_poly) ** (0.5)

    # predict validate
    yval_pred_poly = linear_model_pf.predict(xval_degree2)

    # evaluate: RMSE
    rmse_val_poly = mean_squared_error(yval, yval_pred_poly) ** (0.5)

    # how important is each feature to the target
    polyFeat.coef_


    # Create a DataFrame with a single row
    data = {'RMSE Training': rmse_train_poly,
            'RMSE Validation': rmse_val_poly,
            'RMSE baseline': baseline_mean,
            'train validate RMSE diff': rmse_val_poly - rmse_train_poly,
        'val baseline diff': rmse_val_poly - baseline_mean,
        'R2_validate': explained_variance_score(yval, yval_pred_poly)}
    results = pd.DataFrame([data])
    return results

def get_best_model():
    """
    return best model predictions
    """
    # MAKE THE THING: create the model object
    linear_nodel_twd = TweedieRegressor(alpha= 1.0, power= 1)

    #1. FIT THE THING: fit the model to training data
    # We must specify the column in y_train, since we have converted it to a dataframe from a series!
    tweedieReg = linear_nodel_twd.fit(xtrain, ytrain)

    #2. USE THE THING: make a prediction
    ytrain_pred_twd = linear_nodel_twd.predict(xtrain)

    #3. Evaluate: RMSE
    rmse_train_twd = mean_squared_error(ytrain, ytrain_pred_twd) ** (0.5)

    # predict validate
    ytest_pred_twd = linear_nodel_twd.predict(xtest)

    # evaluate: RMSE
    rmse_test_twd = mean_squared_error(ytest, ytest_pred_twd) ** (0.5)

    # how important is each feature to the target
    tweedieReg.coef_

    # Create a DataFrame with a single row
    data = {'RMSE Training': rmse_train_twd,
            'RMSE test': rmse_test_twd,
            'RMSE baseline': baseline_mean,
            'train test RMSE diff': rmse_test_twd - rmse_train_twd,
        'test baseline diff': rmse_test_twd - baseline_mean,
        'R2 test': explained_variance_score(ytest, ytest_pred_twd)}
    results = pd.DataFrame([data])

    return results

