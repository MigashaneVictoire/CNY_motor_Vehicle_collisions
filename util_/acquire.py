# importing my data
import pandas as pd

def get_data() -> pd.DataFrame:
    """
    Goal: return the original dogs data I need for the project.
    """
    return pd.read_csv("project_data/DOHMH_Dog_Bite_Data.csv")
