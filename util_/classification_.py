# funtion annotations
from binascii import a2b_qp
from typing import Union
from typing import Tuple

# data manipulation
import pandas as pd
import numpy as np

# data visualization
import matplotlib.pyplot as plt
import seaborn as sns
# set a default them for all my visuals
sns.set_theme(style="whitegrid")
plt.ioff() # Turn off interactive mode

# modeling
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import ConfusionMatrixDisplay

# for knn
from sklearn.neighbors import KNeighborsClassifier

# for decision tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree

# for random forest
from sklearn.ensemble import RandomForestClassifier

# for logistic regression
from sklearn.linear_model import LogisticRegression

# system manipulation
from itertools import combinations
import os
import sys
sys.path.append("./util_")
import prepare_
import explore_ 

# other
import math
import env
import warnings
warnings.filterwarnings("ignore")

# set the random seed
np.random.seed(95)

#######################################################################################
################ Baseline accuracy score ----------------------------------------------
#######################################################################################
def baseline_accuracy(target:np.array, base_avg_metrics: int= 0) -> float:
    """
    Goal: establish baseline accuracy score
    perimeters:
        target: numpy array representing our target variable.
        base_avg_metrics: how to evaluate the baseline prediction.
            -> mode = 0
            --> median = 1
    return:
        baseline accuracy score.
    """
    if base_avg_metrics == 0:
        # calculate and add bseline to the training data
        avg_metrics = target.mode()
    elif base_avg_metrics == 1:
         # calculate and add bseline to the training data
        avg_metrics = target.median()

    # baseline score
    baseline =accuracy_score( target, [avg_metrics] * len(target))
    return round(baseline, 3)

#######################################################################################
################ KNN classification model----------------------------------------------
#######################################################################################
# Train KNN model on different k values.
def get_knn_(xtrain:pd.DataFrame, xval:pd.DataFrame,ytrain:np.array, yval:np.array, iter_percent:float=0.005, base_avg_metrics:int= 0) -> Tuple[pd.DataFrame,str, plt.plot]:
    """
    Goal: Perform full k-nearest neighbor mode and return the resuls
    perimeters:
        xtrain/xval: training and validation feature data (pd.DataFrame)
        ytrain/yval: training and validation target variable (np.array)
        iter_percent: percentage of the number of max iteration based on the length of training data
        base_avg_metrics: baseline evaluation method (mode = 0 or median = 1)
    return:
        knn_model_df: pandas data frame of all the models performed.
        cls_report: classification report
        plt.gcf(): visual of the knn_model scores
    """
    # the maximun number of neighbors the model should look at
    # in my case it can only look at 1% of the data
    k_neighbors = math.ceil(len(xtrain) * iter_percent)
    baseline = baseline_accuracy(target=ytrain, base_avg_metrics=base_avg_metrics)

    # the final result metric
    iter_lst = []

    for k in range(1, k_neighbors + 1):
        # create a knn object
        #                          n_neighborsint(default=5) 
        knn = KNeighborsClassifier(n_neighbors=k, weights='uniform', p=2)
        #                                                        p=1 uses the manhattan distance

        # Fit training data to the object
        knn.fit(xtrain, ytrain)
        
        # use the object to make predictons
        ypred_train = knn.predict(xtrain)
        ypred_val = knn.predict(xval)

        # get the prediction scores
        train_score = knn.score(xtrain, ytrain)
        validate_score = knn.score(xval, yval)
        
        # create a dictionary of scores
        one_iteration = {
            "k": k,
            "train_score": train_score,
            "validate_score": validate_score,
            "train_val_difference": train_score - validate_score,
            "train_baseline_diff": baseline - train_score,
            "baseline_accuracy": baseline,
        }
        
        iter_lst.append(one_iteration)

    # create models dataframe
    knn_model_df = pd.DataFrame(iter_lst)

    #PLOT train vs validate
    knn_model_df[knn_model_df.columns[:-3]].set_index("k").plot()
    plt.xticks(np.arange(0,k_neighbors, 2))
    plt.ylabel('accuracy')
    plt.xlabel('n_neighbors')
    plt.title("Train vs Validate Accuracy Scores")
    plt.grid(visible=True, axis='both')

    # get classification report
    cls_report = classification_report(ytrain, ypred_train)

    return knn_model_df, cls_report, plt.gcf()

#######################################################################################
################ Decision Tree Model----------------------------------------------
#######################################################################################
def get_decision_tree_(xtrain:pd.DataFrame, xval:pd.DataFrame,ytrain:np.array, yval:np.array, max_num_depth:int=11, base_avg_metrics:int= 0) -> Tuple[pd.DataFrame, str, plt.plot]:
    """
    Goal: Perform full decision tree mode and return the resuls
    perimeters:
        xtrain/xval: training and validation feature data (pd.DataFrame)
        ytrain/yval: training and validation target variable (np.array)
        max_num_depth: maximum number of trees to loop through
        base_avg_metrics: baseline evaluation method (mode = 0 or median = 1)
    return:
        knn_model_df: pandas data frame of all the models performed.
        cls_report: classification report
        plt.gcf(): visual of the knn_model scores
    """
    baseline = baseline_accuracy(target=ytrain, base_avg_metrics=base_avg_metrics)
    # the final result metric
    iter_lst = []

    for d in range(1,max_num_depth):
    #      create tree object
        treeClf = DecisionTreeClassifier(max_depth= d, random_state=95)
        
        # fit model
        treeClf.fit(xtrain, ytrain)

        # use the object to make predictons
        ypred_train = treeClf.predict(xtrain)
        ypred_val = treeClf.predict(xval)
        
        # train accurecy score
        train_score = treeClf.score(xtrain, ytrain)
        validate_score = treeClf.score(xval, yval)
        
        # create a dictionary of scores
        one_iteration = {
            "depth": d,
            "train_score": train_score,
            "validate_score": validate_score,
            "difference": train_score - validate_score,
            "train_baseline_diff": baseline - train_score,
            "baseline_accuracy": baseline,
        }
        
        iter_lst.append(one_iteration)

    
    # get the result as a dataframe
    decTree_model_df = pd.DataFrame(iter_lst)

    #PLOT train vs validate
    decTree_model_df[decTree_model_df.columns[:-3]].set_index("depth").plot()
    plt.ylabel('accuracy')
    plt.xlabel('tree_depth')
    plt.title("Train vs Validate Accuracy Scores")
    plt.xticks(np.arange(0,max_num_depth))
    plt.grid(visible=True, axis='both')
    
    # get classification report
    cls_report = classification_report(ytrain, ypred_train)

    return decTree_model_df, cls_report, plt.gcf()

#######################################################################################
################ Random Forest Model----------------------------------------------
#######################################################################################

def get_random_forest_(xtrain:pd.DataFrame, xval:pd.DataFrame,ytrain:np.array, yval:np.array, max_num_trees:int= 20,base_avg_metrics:int= 0) -> Tuple[pd.DataFrame, str, plt.plot]:
    """
    Goal: Perform full decision tree mode and return the resuls
    perimeters:
        xtrain/xval: training and validation feature data (pd.DataFrame)
        ytrain/yval: training and validation target variable (np.array)
        max_num_trees: maximum number of trees to loop through
        base_avg_metrics: baseline evaluation method (mode = 0 or median = 1)
    return:
        knn_model_df: pandas data frame of all the models performed.
        cls_report: classification report
        plt.gcf(): visual of the knn_model scores
    """
    baseline = baseline_accuracy(target=ytrain, base_avg_metrics=base_avg_metrics)
    # the final result metric
    iter_lst = []

    for trees in range(2,max_num_trees):

        # create ramdom tree object
        randFor = RandomForestClassifier(n_estimators= 100, min_samples_leaf= trees, max_depth = trees, random_state=95 )
        
        # fit the model
        randFor.fit(xtrain, ytrain)

        # use the object to make predictons
        ypred_train = randFor.predict(xtrain)
        ypred_val = randFor.predict(xval)
        
        # get accuracy scores
        train_score = randFor.score(xtrain, ytrain)
        validate_score = randFor.score(xval, yval)
        
        # create a dictionary of scores
        one_iteration = {
            "trees": trees,
            "train_score": train_score,
            "validate_score": validate_score,
            "difference": train_score - validate_score,
            "train_baseline_diff": baseline - train_score,
            "baseline_accuracy": baseline,
        }
        
        iter_lst.append(one_iteration)

    # get the result as a dataframe
    randFor_model_df = pd.DataFrame(iter_lst)        

    #PLOT train vs validate
    randFor_model_df[randFor_model_df.columns[:-3]].set_index("trees").plot()
    plt.ylabel('accuracy')
    plt.xlabel('tree_depth')
    plt.title("Train vs Validate Accuracy Scores")
    plt.xticks(np.arange(2,max_num_trees))
    plt.grid(visible=True, axis='both')

    # get classification report
    cls_report = classification_report(ytrain, ypred_train)

    return randFor_model_df, cls_report, plt.gcf()

#######################################################################################
################ Logistic Regression Model--------------------------------------------
#######################################################################################
def get_logistic_regression_(xtrain:pd.DataFrame, xval:pd.DataFrame,ytrain:np.array, yval:np.array, max_num_iter:float=0.01,base_avg_metrics:int= 0) -> Tuple[pd.DataFrame, str, plt.plot]:
    """
    Goal: Perform full decision tree mode and return the resuls
    perimeters:
        xtrain/xval: training and validation feature data (pd.DataFrame)
        ytrain/yval: training and validation target variable (np.array)
        max_num_iter: maximum number of iterations to loop through
        base_avg_metrics: baseline evaluation method (mode = 0 or median = 1)
    return:
        knn_model_df: pandas data frame of all the models performed.
        cls_report: classification report
        plt.gcf(): visual of the knn_model scores
    """
    baseline = baseline_accuracy(target=ytrain, base_avg_metrics=base_avg_metrics)
    # the final result metric
    iter_lst = []

    for c in np.arange(0.0001,0.1, max_num_iter):
        
        # create ramdom tree object
        logReg = LogisticRegression(C=c, random_state=95, max_iter= 1000)
        
        # fit the model
        logReg.fit(xtrain, ytrain)

        # use the object to make predictons
        ypred_train = logReg.predict(xtrain)
        ypred_val = logReg.predict(xval)

        # get accuracy scores
        train_score = logReg.score(xtrain, ytrain)
        validate_score = logReg.score(xval, yval)
        
        # create a dictionary of scores
        one_iteration = {
            "c": c,
            "train_score": train_score,
            "validate_score": validate_score,
            "difference": train_score - validate_score,
            "train_baseline_diff": baseline - train_score,
            "baseline_accuracy": baseline,
        }
        
        iter_lst.append(one_iteration)
        
    # get the result as a dataframe
    logReg_model_df = pd.DataFrame(iter_lst)

    #PLOT train vs validate
    logReg_model_df[logReg_model_df.columns[:-3]].set_index("c").plot()
    plt.ylabel('accuracy')
    plt.xlabel('C')
    plt.title("Train vs Validate Accuracy Scores")
    plt.grid(visible=True, axis='both')

    # get classification report
    cls_report = classification_report(ytrain, ypred_train)

    return logReg_model_df, cls_report, plt.gcf()

#######################################################################################
################ Best Model test--------------------------------------------------------
#######################################################################################
def get_test_best_model_(xtrain:pd.DataFrame, xtest:pd.DataFrame,ytrain:np.array, ytest:np.array, best_model:int=None, k_depth_tree_c_=None, base_avg_metrics:int= 0) -> Tuple[pd.DataFrame, str, plt.plot]:
    """
    Goal: Perform full decision tree mode and return the resuls
    perimeters:
        xtrain/xval: training and validation feature data (pd.DataFrame)
        ytrain/yval: training and validation target variable (np.array)
        k_depth_tree_c_: The best [k, depth, tree, or c] found from training and validation stages
        base_avg_metrics: baseline evaluation method (mode = 0 or median = 1)
    return:
        knn_model_df: pandas data frame of all the models performed.
        cls_report: classification report
        plt.gcf(): visual of the knn_model scores
    """
    baseline = baseline_accuracy(target=ytrain, base_avg_metrics=base_avg_metrics)

    # knn
    if best_model == 1:
        iter = "k"

        # create a knn object
        #                          n_neighborsint(default=5) 
        knn = KNeighborsClassifier(n_neighbors=k_depth_tree_c_, weights='uniform', p=2)
        #                                                        p=1 uses the manhattan distance

        # Fit training data to the object
        knn.fit(xtrain, ytrain)

        # get the prediction scores
        train_score = knn.score(xtrain, ytrain)
        test_score = knn.score(xtest, ytest)
    
    # Decission tree
    elif best_model == 2:
        iter = "depth"

        # create tree object
        treeClf = DecisionTreeClassifier(max_depth= k_depth_tree_c_, random_state=95)

        # fit model
        treeClf.fit(xtrain, ytrain)

        # train accurecy score
        train_score = treeClf.score(xtrain, ytrain)
        test_score = treeClf.score(xtest, ytest)

    # Random forest
    elif best_model == 3:
        iter = "trees"

        # create ramdom tree object
        randFor = RandomForestClassifier(n_estimators= 100, min_samples_leaf= k_depth_tree_c_, max_depth = k_depth_tree_c_, random_state=95 )
        
        # fit the model
        randFor.fit(xtrain, ytrain)
        
        # get accuracy scores
        train_score = randFor.score(xtrain, ytrain)
        test_score = randFor.score(xtest, ytest)
        
    # Logistic regression
    elif best_model == 4:
        iter = "c"

        # create ramdom tree object
        logReg = LogisticRegression(C=k_depth_tree_c_, random_state=95, max_iter= 1000)
        
        # fit the model
        logReg.fit(xtrain, ytrain)

        # get accuracy scores
        train_score = logReg.score(xtrain, ytrain)
        test_score = logReg.score(xtest, ytest)
    else:
        return "You need to pass a valid value (k, depth, tree, or c) for your best performing model"

    # create a dictionary of scores
    output = {
        f"{iter}": k_depth_tree_c_,
        "train_score": train_score,
        "test_score": test_score,
        "train_test_diff": train_score - test_score,
        "test_baseline_diff": baseline - test_score,
        "baseline_accuracy": baseline,
    }
    # get the result as a dataframe
    classification_best_model = pd.DataFrame([output])
    return classification_best_model