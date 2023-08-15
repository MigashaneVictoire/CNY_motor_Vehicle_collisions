# data manipulation
import pandas as pd
import numpy as np

# data visualization
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

# data separation/transformation
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE # Recursive Feature Elimination¶

# modeling
from sklearn.cluster import KMeans
import sklearn.preprocessing

# statistics testing
import scipy.stats as stats

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

##############################################################################
################### Cluster 
##############################################################################
def get_cluster_(train:pd.DataFrame,feature_1:np.array, feature_2:np.array, iter_percent:float=5.005):
    """
    """
    # set the max number ok of k to loop through
    # 5% of the data
    iter = math.ceil(len(feature_1) * iter_percent)

    model_centers = []
    model_inertia = []

    for k in range(1,iter + 1):
        # ceate model object
        kmean = KMeans(n_clusters= k)

        # fit model object
        kmean.fit(train_scaled[list(feature_combinations[0])])

        # make predictions
        label = kmean.predict(train_scaled[list(feature_combinations[0])])
        
        # add predictions to the original dataframe
        model_df[f"clusters_{k}"] = label

        # view ceters
        model_centers.append(kmean.cluster_centers_)
        
        model_inertia.append(kmean.inertia_)
    return 