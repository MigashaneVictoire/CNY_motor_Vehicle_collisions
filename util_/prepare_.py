# For funtion annotations
from binascii import a2b_qp
from typing import Union
from typing import Tuple

# Python libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import os

# Personal libraries
import acquire_
# import env

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

# -----------------------------------------------------------------
# get data without dummies
def get_vehicle_data_no_dummies() -> pd.DataFrame:
    """
    return prepared vehicle data that doesn't contain dummie variables
    """
    # get data from acquire file
    vehicle = acquire_.get_data()
    
    # 1. make every columns lower case
    # 2. replace all the spaces with inderscore
    vehicle = vehicle.rename(columns=lambda x: x.lower().replace(" ", "_"))

    # convert string to datetime
    vehicle["crash_date"] = pd.to_datetime(vehicle['crash_date'])
    vehicle["crash_time"] = pd.to_datetime(vehicle['crash_time'])

    # removeing all the null columns for number_of_persons_injured and number_of_persons_killed
    vehicle = vehicle[vehicle.number_of_persons_injured.notna()]
    vehicle = vehicle[vehicle.number_of_persons_killed.notna()]

    # change column data type from float to int
    vehicle.number_of_persons_injured = vehicle.number_of_persons_injured.astype("int")
    vehicle.number_of_persons_killed = vehicle.number_of_persons_killed.astype("int")

    # replace NaN with 00000 to signify unknown zipcode
    vehicle.zip_code = vehicle.zip_code.fillna("0")

    # replace empty strings with 0 to signify unknown zipcode
    vehicle.zip_code = vehicle.zip_code.str.replace(" ", "0")

    # make zipcode column numeric
    vehicle.zip_code = vehicle.zip_code.astype("int")

    # replace all None in object columns with Unknown
    vehicle[vehicle.select_dtypes("object").columns] = vehicle.select_dtypes("object").fillna("UNKNOWN")

    # Group similar factors together   
    factors_1 = {
        "Visibility and Road Conditions":["Windshield Inadequate",
                                        "Headlights Defective",
                                        "Other Lighting Defects",
                                        "Glare",
                                        "View Obstructed/Limited",
                                        "Pavement Slippery",
                                        "Obstruction/Debris",
                                        "Pavement Defective"],
        "Distractions from Electronic Devices":["Cell Phone (hand-Held)", "Cell Phone (hand-held)",
                                                "Cell Phone (hands-free)",
                                                "Other Electronic Device",
                                                "Outside Car Distraction"],
        "Impairment (Alcohol, Drugs, Medication)":["Alcohol Involvement",
                                                "Drugs (illegal)","Drugs (Illegal)",
                                                "Prescription Medication"],
        "Driver Fatigue and Inattention":["Fell Asleep",
                                        "Lost Consciousness",
                                        "Fatigued/Drowsy",
                                        "Illnes", "Illness",
                                        "Unsafe Speed",
                                        "Driver Inattention/Distraction",
                                        "80"],
        "Unsafe Driving Maneuvers":["Unsafe Lane Changing",
                                    "Passing Too Closely",
                                    "Turning Improperly",
                                    "Passing or Lane Usage Improper",
                                    "Failure to Yield Right-of-Way",
                                    "Failure to Keep Right"],
        "Vehicle Equipment Failure":["Following Too Closely",
                                    "Traffic Control Disregarded",
                                    "Accelerator Defective",
                                    "Brakes Defective",
                                    "Steering Failure",
                                    "Tire Failure/Inadequate"],
        "Issues with Traffic Control and Lane Marking":["Traffic Control Device Improper/Non-Working",
                                                        "Lane Marking Improper/Inadequate"],
        "Driver Characteristics and Experience":["Physical Disability",
                                                "Driver Inexperience"],
        "Reactions to Other Vehicles":["Reaction to Other Uninvolved Vehicle",
                                    "Reaction to Uninvolved Vehicle"],
        "Distracted Driving":["Listening/Using Headphones",
                            "Texting",
                            "Eating or Drinking",
                            "Distracted Driving"],
        "Vehicle-related Incidents":["Vehicle Vandalism",
                                    "Tow Hitch Defective",
                                    "Driverless/Runaway Vehicle",
                                    "Oversized Vehicle",
                                    "Other Vehicular"],
        "Interactions with Pedestrians and Cyclists":["Animals Action",
                                                    "Pedestrian/Bicyclist/Other Pedestrian Error/Confusion"],
        "Aggressive Driving and Passenger Distraction":["Aggressive Driving/Road Rage",
                                                        "Passenger Distraction"],
        "Unsafe Lane Changes and Backing":["Unsafe Lane Changing",
                                        "Passing Too Closely",
                                        "Turning Improperly",
                                        "Passing or Lane Usage Improper",
                                        "Backing Unsafely"],
        "Other":["Using On Board Navigation Device",
                "Tinted Windows",
                "Shoulders Defective/Improper"],
        "Uncertain or Unspecified Factors":["UNKNOWN",
                                            "Unspecified",
                                        "1",
                                        ""]
    }

    # Re-assigning new names to the feature items.
    for k, v in factors_1.items(): # iterate trough the keys and values of the dictionary
        for ele in v: # iterate throug only the values 
            # Replace the entire cell with 'replacement_value' if a match is found
            vehicle["contributing_factor_vehicle_1"] = vehicle["contributing_factor_vehicle_1"].apply(lambda x: k if x == ele else x)
            vehicle["contributing_factor_vehicle_2"] = vehicle["contributing_factor_vehicle_2"].apply(lambda x: k if x == ele else x)
            vehicle["contributing_factor_vehicle_3"] = vehicle["contributing_factor_vehicle_3"].apply(lambda x: k if x == ele else x)
            vehicle["contributing_factor_vehicle_4"] = vehicle["contributing_factor_vehicle_4"].apply(lambda x: k if x == ele else x)
            vehicle["contributing_factor_vehicle_5"] = vehicle["contributing_factor_vehicle_5"].apply(lambda x: k if x == ele else x)
   
   # Group the categories
    vehicle_categories = {
        'SEDAN': ['Sedan', '4 dr sedan', '2 dr sedan', '3-Door'],
        'STATION_WAGON': ['Station Wagon/Sport Utility Vehicle', 'SPORT UTILITY / STATION WAGON'],
        'PASSENGER_VEHICLE': ['PASSENGER VEHICLE'],
        'TAXI': ['Taxi', 'TAXI'],
        'PICKUP_TRUCK': ['Pick-up Truck', 'PICK-UP TRUCK', 'PK', 'Pickup with mounted Camper', 'pick'],
        'UNKNOWN': ['UNKNOWN', 'UNKNO', 'UNK','unknown', 'unko', 'unk'],
        'VAN': ['VAN', 'van', 'Van', 'Van Camper'],
        'BOX_TRUCK': ['Box Truck', 'BOX T', 'BOX TRUCK'],
        'BUS': ['Bus', 'BUS', 'School Bus'],
        'LARGE_COM_VEH': ['LARGE COM VEH(6 OR MORE TIRES)'],
        'SMALL_COM_VEH': ['SMALL COM VEH(4 TIRES)', 'COMME'],
        'LIVERY_VEHICLE': ['LIVERY VEHICLE'],
        'TRACTOR_TRUCK_DIESEL': ['Tractor Truck Diesel', 'TRACT'],
        'MOTORCYCLE': ['Bike', 'MOTORCYCLE', 'Motorscooter', 'Moped', 'Minibike', 'Minicycle'],
        'AMBULANCE': ['Ambulance', 'AMBULANCE', 'AMBUL', 'ambul', 'AMB', 'FDNY AMBUL', 'fdny'],
        'CONVERTIBLE': ['Convertible'],
        'DUMP': ['Dump', 'dump'],
        'E_BIKE': ['E-Bike', 'E-Bik', 'ELECT'],
        'FLAT_BED': ['Flat Bed', 'FLAT'],
        'GARBAGE_OR_REFUSE': ['Garbage or Refuse'],
        'CARRY_ALL': ['Carry All'],
        'E_SCOOTER': ['E-Scooter', 'E-Sco'],
        'TRACTOR_TRUCK_GASOLINE': ['Tractor Truck Gasoline', 'FORD'],
        'TOW_TRUCK': ['Tow Truck / Wrecker', 'Tow Truck', 'TOW T'],
        'FIRE_TRUCK': ['FIRE TRUCK', 'FIRET', 'Fire Truck', 'fire', 'FDNY FIRE', 'FDNY TRUCK', 'FDNY'],
        'CHASSIS_CAB': ['Chassis Cab'],
        'TANKER': ['Tanker', 'TANK'],
        'REFRIGERATED_VAN': ['Refrigerated Van'],
        'CONCRETE_MIXER': ['Concrete Mixer'],
        'FLAT_RACK': ['Flat Rack'],
        'ARMORED_TRUCK': ['Armored Truck'],
        'BEVERAGE_TRUCK': ['Beverage Truck'],
        'SCOOTER': ['SCOOTER', 'SCOOT'],
        'LIMO': ['LIMO'],
        'LIFT_BOOM': ['Lift Boom'],
        'TRUCK': ['TRUCK', 'truck'],
        'TRAILER': ['TRAIL', 'trail', 'TRAILER'],
        'STAKE_OR_RACK': ['Stake or Rack'],
        'LUNCH_WAGON': ['Lunch Wagon'],
        'FORKLIFT': ['FORKL'],
        'MOTORIZED_HOME': ['Motorized Home'],
        'PEDICAB': ['Pedicab'],
        'HOPPER': ['Hopper'],
        'MULTI_WHEELED_VEHICLE': ['Multi-Wheeled Vehicle'],
        'USPS': ['USPS'],
        'DELIVERY': ['DELIV', 'DELV'],
        'UTILITY': ['UTILI', 'UTIL'],
        'OPEN_BODY': ['Open Body'],
        'BULK_AGRICULTURE': ['Bulk Agriculture']
    }
    

    # Replacing values in the "vehicle_type_code_1" column based on the categories
    def replace_category(value):
        for category, codes in vehicle_categories.items():
            if value in codes:
                return category
        return 'OTHER'

    # apply the created function to the data
    vehicle["vehicle_type_code_1"] = vehicle["vehicle_type_code_1"].apply(replace_category)
    vehicle["vehicle_type_code_2"] = vehicle["vehicle_type_code_2"].apply(replace_category)
    vehicle["vehicle_type_code_3"] = vehicle["vehicle_type_code_3"].apply(replace_category)
    vehicle["vehicle_type_code_4"] = vehicle["vehicle_type_code_4"].apply(replace_category)
    vehicle["vehicle_type_code_5"] = vehicle["vehicle_type_code_5"].apply(replace_category)

    # if any fatalities are found add 1 in the list
    fatalities = []
    for row in range(len(vehicle)):
        # locate each row and check if any number of fatalities accured
        if vehicle.number_of_persons_killed.iloc[row] != 0:
            fatalities.append(1)
        elif vehicle.number_of_pedestrians_killed.iloc[row] != 0:
            fatalities.append(1)
        elif vehicle.number_of_cyclist_killed.iloc[row] != 0:
            fatalities.append(1)
        elif vehicle.number_of_motorist_killed.iloc[row] != 0:
            fatalities.append(1)
        else:
            fatalities.append(0)

    # add target variable to the data
    vehicle["fatality"] = fatalities

    # drop redundent columns
    remove_cols = ["location", "collision_id", 
                "number_of_persons_killed",
                "number_of_pedestrians_killed",
                "number_of_cyclist_killed",
                "number_of_motorist_killed"]
    vehicle = drop_cols(vehicle, remove_cols)    

    return vehicle

# -----------------------------------------------------------------
# get data with dummies
def get_vehicle_data_with_dummies() -> pd.DataFrame:
    """
    return prepared training data that does contain dummie variables
    """
    # get data from acquire file
    vehicle = acquire_.get_data()
    
    # 1. make every columns lower case
    # 2. replace all the spaces with inderscore
    vehicle = vehicle.rename(columns=lambda x: x.lower().replace(" ", "_"))

    # convert string to datetime
    vehicle["crash_date"] = pd.to_datetime(vehicle['crash_date'])
    vehicle["crash_time"] = pd.to_datetime(vehicle['crash_time'])

    # removeing all the null columns for number_of_persons_injured and number_of_persons_killed
    vehicle = vehicle[vehicle.number_of_persons_injured.notna()]
    vehicle = vehicle[vehicle.number_of_persons_killed.notna()]

    # change column data type from float to int
    vehicle.number_of_persons_injured = vehicle.number_of_persons_injured.astype("int")
    vehicle.number_of_persons_killed = vehicle.number_of_persons_killed.astype("int")

    # replace NaN with 00000 to signify unknown zipcode
    vehicle.zip_code = vehicle.zip_code.fillna("0")

    # replace empty strings with 0 to signify unknown zipcode
    vehicle.zip_code = vehicle.zip_code.str.replace(" ", "0")

    # make zipcode column numeric
    vehicle.zip_code = vehicle.zip_code.astype("int")

    # replace all None in object columns with Unknown
    vehicle[vehicle.select_dtypes("object").columns] = vehicle.select_dtypes("object").fillna("UNKNOWN")

    # Group similar factors together   
    factors_1 = {
        "Visibility and Road Conditions":["Windshield Inadequate",
                                        "Headlights Defective",
                                        "Other Lighting Defects",
                                        "Glare",
                                        "View Obstructed/Limited",
                                        "Pavement Slippery",
                                        "Obstruction/Debris",
                                        "Pavement Defective"],
        "Distractions from Electronic Devices":["Cell Phone (hand-Held)", "Cell Phone (hand-held)",
                                                "Cell Phone (hands-free)",
                                                "Other Electronic Device",
                                                "Outside Car Distraction"],
        "Impairment (Alcohol, Drugs, Medication)":["Alcohol Involvement",
                                                "Drugs (illegal)","Drugs (Illegal)",
                                                "Prescription Medication"],
        "Driver Fatigue and Inattention":["Fell Asleep",
                                        "Lost Consciousness",
                                        "Fatigued/Drowsy",
                                        "Illnes", "Illness",
                                        "Unsafe Speed",
                                        "Driver Inattention/Distraction",
                                        "80"],
        "Unsafe Driving Maneuvers":["Unsafe Lane Changing",
                                    "Passing Too Closely",
                                    "Turning Improperly",
                                    "Passing or Lane Usage Improper",
                                    "Failure to Yield Right-of-Way",
                                    "Failure to Keep Right"],
        "Vehicle Equipment Failure":["Following Too Closely",
                                    "Traffic Control Disregarded",
                                    "Accelerator Defective",
                                    "Brakes Defective",
                                    "Steering Failure",
                                    "Tire Failure/Inadequate"],
        "Issues with Traffic Control and Lane Marking":["Traffic Control Device Improper/Non-Working",
                                                        "Lane Marking Improper/Inadequate"],
        "Driver Characteristics and Experience":["Physical Disability",
                                                "Driver Inexperience"],
        "Reactions to Other Vehicles":["Reaction to Other Uninvolved Vehicle",
                                    "Reaction to Uninvolved Vehicle"],
        "Distracted Driving":["Listening/Using Headphones",
                            "Texting",
                            "Eating or Drinking",
                            "Distracted Driving"],
        "Vehicle-related Incidents":["Vehicle Vandalism",
                                    "Tow Hitch Defective",
                                    "Driverless/Runaway Vehicle",
                                    "Oversized Vehicle",
                                    "Other Vehicular"],
        "Interactions with Pedestrians and Cyclists":["Animals Action",
                                                    "Pedestrian/Bicyclist/Other Pedestrian Error/Confusion"],
        "Aggressive Driving and Passenger Distraction":["Aggressive Driving/Road Rage",
                                                        "Passenger Distraction"],
        "Unsafe Lane Changes and Backing":["Unsafe Lane Changing",
                                        "Passing Too Closely",
                                        "Turning Improperly",
                                        "Passing or Lane Usage Improper",
                                        "Backing Unsafely"],
        "Other":["Using On Board Navigation Device",
                "Tinted Windows",
                "Shoulders Defective/Improper"],
        "Uncertain or Unspecified Factors":["UNKNOWN",
                                            "Unspecified",
                                        "1",
                                        ""]
    }

    # Re-assigning new names to the feature items.
    for k, v in factors_1.items(): # iterate trough the keys and values of the dictionary
        for ele in v: # iterate throug only the values 
            # Replace the entire cell with 'replacement_value' if a match is found
            vehicle["contributing_factor_vehicle_1"] = vehicle["contributing_factor_vehicle_1"].apply(lambda x: k if x == ele else x)
            vehicle["contributing_factor_vehicle_2"] = vehicle["contributing_factor_vehicle_2"].apply(lambda x: k if x == ele else x)
            vehicle["contributing_factor_vehicle_3"] = vehicle["contributing_factor_vehicle_3"].apply(lambda x: k if x == ele else x)
            vehicle["contributing_factor_vehicle_4"] = vehicle["contributing_factor_vehicle_4"].apply(lambda x: k if x == ele else x)
            vehicle["contributing_factor_vehicle_5"] = vehicle["contributing_factor_vehicle_5"].apply(lambda x: k if x == ele else x)

    # Group the categories
    vehicle_categories = {
        'SEDAN': ['Sedan', '4 dr sedan', '2 dr sedan', '3-Door'],
        'STATION_WAGON': ['Station Wagon/Sport Utility Vehicle', 'SPORT UTILITY / STATION WAGON'],
        'PASSENGER_VEHICLE': ['PASSENGER VEHICLE'],
        'TAXI': ['Taxi', 'TAXI'],
        'PICKUP_TRUCK': ['Pick-up Truck', 'PICK-UP TRUCK', 'PK', 'Pickup with mounted Camper', 'pick'],
        'UNKNOWN': ['UNKNOWN', 'UNKNO', 'UNK','unknown', 'unko', 'unk'],
        'VAN': ['VAN', 'van', 'Van', 'Van Camper'],
        'BOX_TRUCK': ['Box Truck', 'BOX T', 'BOX TRUCK'],
        'BUS': ['Bus', 'BUS', 'School Bus'],
        'LARGE_COM_VEH': ['LARGE COM VEH(6 OR MORE TIRES)'],
        'SMALL_COM_VEH': ['SMALL COM VEH(4 TIRES)', 'COMME'],
        'LIVERY_VEHICLE': ['LIVERY VEHICLE'],
        'TRACTOR_TRUCK_DIESEL': ['Tractor Truck Diesel', 'TRACT'],
        'MOTORCYCLE': ['Bike', 'MOTORCYCLE', 'Motorscooter', 'Moped', 'Minibike', 'Minicycle'],
        'AMBULANCE': ['Ambulance', 'AMBULANCE', 'AMBUL', 'ambul', 'AMB', 'FDNY AMBUL', 'fdny'],
        'CONVERTIBLE': ['Convertible'],
        'DUMP': ['Dump', 'dump'],
        'E_BIKE': ['E-Bike', 'E-Bik', 'ELECT'],
        'FLAT_BED': ['Flat Bed', 'FLAT'],
        'GARBAGE_OR_REFUSE': ['Garbage or Refuse'],
        'CARRY_ALL': ['Carry All'],
        'E_SCOOTER': ['E-Scooter', 'E-Sco'],
        'TRACTOR_TRUCK_GASOLINE': ['Tractor Truck Gasoline', 'FORD'],
        'TOW_TRUCK': ['Tow Truck / Wrecker', 'Tow Truck', 'TOW T'],
        'FIRE_TRUCK': ['FIRE TRUCK', 'FIRET', 'Fire Truck', 'fire', 'FDNY FIRE', 'FDNY TRUCK', 'FDNY'],
        'CHASSIS_CAB': ['Chassis Cab'],
        'TANKER': ['Tanker', 'TANK'],
        'REFRIGERATED_VAN': ['Refrigerated Van'],
        'CONCRETE_MIXER': ['Concrete Mixer'],
        'FLAT_RACK': ['Flat Rack'],
        'ARMORED_TRUCK': ['Armored Truck'],
        'BEVERAGE_TRUCK': ['Beverage Truck'],
        'SCOOTER': ['SCOOTER', 'SCOOT'],
        'LIMO': ['LIMO'],
        'LIFT_BOOM': ['Lift Boom'],
        'TRUCK': ['TRUCK', 'truck'],
        'TRAILER': ['TRAIL', 'trail', 'TRAILER'],
        'STAKE_OR_RACK': ['Stake or Rack'],
        'LUNCH_WAGON': ['Lunch Wagon'],
        'FORKLIFT': ['FORKL'],
        'MOTORIZED_HOME': ['Motorized Home'],
        'PEDICAB': ['Pedicab'],
        'HOPPER': ['Hopper'],
        'MULTI_WHEELED_VEHICLE': ['Multi-Wheeled Vehicle'],
        'USPS': ['USPS'],
        'DELIVERY': ['DELIV', 'DELV'],
        'UTILITY': ['UTILI', 'UTIL'],
        'OPEN_BODY': ['Open Body'],
        'BULK_AGRICULTURE': ['Bulk Agriculture']
    }
        

    # Replacing values in the "vehicle_type_code_1" column based on the categories
    def replace_category(value):
        for category, codes in vehicle_categories.items():
            if value in codes:
                return category
        return 'OTHER'

    # apply the created function to the data
    vehicle["vehicle_type_code_1"] = vehicle["vehicle_type_code_1"].apply(replace_category)
    vehicle["vehicle_type_code_2"] = vehicle["vehicle_type_code_2"].apply(replace_category)
    vehicle["vehicle_type_code_3"] = vehicle["vehicle_type_code_3"].apply(replace_category)
    vehicle["vehicle_type_code_4"] = vehicle["vehicle_type_code_4"].apply(replace_category)
    vehicle["vehicle_type_code_5"] = vehicle["vehicle_type_code_5"].apply(replace_category)


    # if any fatalities are found add 1 in the list
    fatalities = []
    for row in range(len(vehicle)):
        # locate each row and check if any number of fatalities accured
        if vehicle.number_of_persons_killed.iloc[row] != 0:
            fatalities.append(1)
        elif vehicle.number_of_pedestrians_killed.iloc[row] != 0:
            fatalities.append(1)
        elif vehicle.number_of_cyclist_killed.iloc[row] != 0:
            fatalities.append(1)
        elif vehicle.number_of_motorist_killed.iloc[row] != 0:
            fatalities.append(1)
        else:
            fatalities.append(0)

    # add target variable to the data
    vehicle["fatality"] = fatalities

    # drop redundent columns
    remove_cols = ["location", "collision_id", 
                "number_of_persons_killed",
                "number_of_pedestrians_killed",
                "number_of_cyclist_killed",
                "number_of_motorist_killed"]
    vehicle = drop_cols(vehicle, remove_cols)

    # get all columns from dataframe
    all_columns = vehicle.columns

    # containers of different variable types
    categorical = []

    # separate variables
    for col in all_columns:
        # count number of unique valus in the column
        len_of_uniq = len(vehicle[col].unique())
        
        # also checking for only object data types
        if (col != "fatality") and (len_of_uniq <= 20) and (vehicle[col].dtype == "O"):
            categorical.append(col)
        else: pass

    # create dummies of the categorical columns
    dummies = pd.get_dummies(vehicle[categorical])

    # renmae the dummie columns
    dummies = dummies.rename(columns=lambda x: x.lower().replace("-", "_"))

    # add dummies to the dataset
    vehicle[dummies.columns] = dummies

    train, validate, test = split_data_(df= vehicle, 
                                        test_size=0.2, # 20% in the test set
                                        random_state=95,
                                        stratify_col = "fatality")
    
    # using the function to same the files
    save_split_data(vehicle, train, validate, test)

    return train


# -----------------------------------------------------------------
# Save the splited data into separate csv files
def save_split_data(encoded_df: pd.DataFrame, train:pd.DataFrame, validate:pd.DataFrame, test:pd.DataFrame, folder_path: str = "./project_data") -> str:
    """
    parameters:
        encoded_df: full project dataframe that contains the (encoded columns or scalling)
        train: training data set that has been split from the original
        validate: validation data set that has been split from the original
        test: testing data set that has been split from the original
        folder_path: folder path where to save the data sets
    return:
        string to show succes of saving the data
    """
    # create new folder if folder don't aready exist
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        # save the dataframe with dummies in a csv for easy access
        encoded_df.to_csv(f"./{folder_path}/encoded_data.csv", mode="w")

        # save training data
        train.to_csv(f"./{folder_path}/training_data.csv", mode="w")

        # save validate
        validate.to_csv(f"./{folder_path}/validation_data.csv", mode="w")

        # Save test
        test.to_csv(f"./{folder_path}/testing_data.csv", mode="w")

    else:
        # save the dataframe with dummies in a csv for easy access
        encoded_df.to_csv(f"./{folder_path}/encoded_data.csv", mode="w")

        # save training data
        train.to_csv(f"./{folder_path}/training_data.csv", mode="w")

        # save validate
        validate.to_csv(f"./{folder_path}/validation_data.csv", mode="w")

        # Save test
        test.to_csv(f"./{folder_path}/testing_data.csv", mode="w")

    return "Four data sets saved as .csv"



# -----------------------------------------------------------------
# Split the data into train, validate and train
def split_data_(df: pd.DataFrame, test_size: float =.2, validate_size: float =.2, stratify_col: str =None, random_state: int=95) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    '''
    parameters:
        df: pandas dataframe you wish to split
        test_size: size of your test dataset
        validate_size: size of your validation dataset
        stratify_col: the column to do the stratification on
        random_state: random seed for the data

    return:
        train, validate, test DataFrames
    '''
    # no stratification
    if stratify_col == None:
        # split test data
        train_validate, test = train_test_split(df, 
                                                test_size=test_size, 
                                                random_state=random_state)
        # split validate data
        train, validate = train_test_split(train_validate, 
                                           test_size=validate_size/(1-test_size),
                                            random_state=random_state)
    # stratify split
    else:
        # split test data
        train_validate, test = train_test_split(df,
                                                test_size=test_size,
                                                random_state=random_state, 
                                                stratify=df[stratify_col])
        # split validate data
        train, validate = train_test_split(train_validate, 
                                           test_size=validate_size/(1-test_size),
                                           random_state=random_state, 
                                           stratify=train_validate[stratify_col])
    return train, validate, test