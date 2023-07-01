import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# import sys
# sys.path.append("./util_")
# import acquire

def drop_cols(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """
    Goal: drop redundent columns from the data

    perimeters:
        df: pandas datafame to remove columns from.
        cols: list containg all the columns to remove from the data.
    
    return:
        original dataframe with removed columns
    """
    # remove the columns
    print("Original dataframe size:", df.shape)
    df = df.drop(columns=cols)
    print("New dataframe size:", df.shape)
    
    return df