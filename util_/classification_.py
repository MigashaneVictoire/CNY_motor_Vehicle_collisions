# data manipulation
import pandas as pd
import numpy as np

# data visualization
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
# set a default them for all my visuals
# sns.set_theme(style="whitegrid")

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


def baseline_accuracy(train: pd.DataFrame ):
    """
    Goal: establish_baseline accuracy score
    """
    # calculate and add bseline to the training data
    train["baseline_"] = int(ytrain.mode())

    # baseline score
    baseline =accuracy_score( ytrain, train_scaled.baseline)
    baseline
    return


def get_knn_():
    """
    Goal: Perform full k-nearest neighbor mode and return the resuls
    """
    return

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