# importing my data
import pandas as pd

def get_data() -> pd.DataFrame:
    """
    Goal: retreive the original vehicle collision data I need for the project.
    """
    # low memory set to false beacuse some columns have unsoecified datatypes
    return pd.read_csv("CNY_motor_Vehicle_Collisions/Motor_Vehicle_Collisions_-_Crashes.csv", low_memory=False)

