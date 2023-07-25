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
def get_knn_(xtrain:pd.DataFrame, xval:pd.DataFrame,ytrain:np.array, yval:np.array, iter_percent:float=0.005, base_avg_metrics:int= 0) -> Tuple[pd.DataFrame, plt.plot, plt.plot, str]:
    """
    Goal: Perform full k-nearest neighbor mode and return the resuls
    perimeters:
        xtrain/xval: training and validation feature data (pd.DataFrame)
        ytrain/yval: training and validation target variable (np.array)
        iter_percent: percentage of the number of max iteration based on the length of training data
        base_avg_metrics: baseline evaluation method (mode = 0 or median = 1)
    return:
        knn_model_df: pandas data frame of all the models performed.
        plt.gcf(): visual of the knn_model scores
        matrixDisp: visual matrix distplay
        cls_report: classification report
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
        knn = knn.fit(xtrain, ytrain)
        
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
################ KNN classification model----------------------------------------------
#######################################################################################
def get_decision_tree_():
    """
    Goal: Perform full k-nearest neighbor mode and return the resuls
    """
    return

def get_random_forest_():
    """
    Goal: Perform full k-nearest neighbor mode and return the resuls
    """
    return


def get_logistic_regression_():
    """
    Goal: Perform full k-nearest neighbor mode and return the resuls
    """
    return