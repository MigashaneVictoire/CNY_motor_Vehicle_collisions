<a name="top"></a>

# City of New York Motor Vehicle Collision Analysis

by Victoire Migashane

## Description

This project aims to analyze motor vehicle collision data in the City of New York to gain insights into traffic accidents. By exploring historical accident data, I intend to help authorities identify high-risk areas, contributing factors, and trends. This information can be used to implement targeted safety measures and interventions, ultimately leading to a reduction in accidents and improved road safety.

## Goals

The primary objectives of this project were to:

- Create an Interactive Dashboard for Exploring Traffic Accident Data.

## Acquire Data

I acquired the data from [catalog.data.gov](https://catalog.data.gov/dataset/motor-vehicle-collisions-crashes) and performed the following steps:

- The dataset consists of 2,014,469 rows and 29 columns.
- Out of the 29 columns, 11 are numeric, and 18 are string object columns.
- Some null values were present in the dataset.
- I performed descriptive statistics analysis on the data.

## Prepare Data

Data preparation involved the following steps:

- Renamed all columns by replacing empty spaces with underscores and making them all lowercase.
- Removed null values in location data (e.g., zipcode, longitude, etc.).
- Changed column data types:
    - Converted object columns to datetime: `crash_date` and `crash_time`.
- Filled all other nulls with "UNKNOWN" for object data types.
- Grouped similar categories to reduce the size of unique features in columns like `contributing_factor_vehicle` and `vehicle_type_code`.
- Created a target variable as a binary column using:
    - `number_of_persons_killed`, `number_of_pedestrians_killed`, `number_of_cyclist_killed`, and `number_of_motorist_killed`.
- Encoded all categorical columns, resulting in 299 new dummy columns.
- Split the data into training, validation, and test sets using a 60/20/20 ratio.

## Data Exploration (EDA)

Data was split into training, validation, and test datasets to avoid bias and data leakage during modeling. Additional exploration questions included:

## Explore

- Seasonal trends were found in the number of motorists, cyclists, and personnel injured. However, there wasn't a strong seasonal trend in the number of pedestrians injured and fatalities.
- Brooklyn had the highest number of collisions (250,000), while Staten Island had the lowest count (about 30,000).
- Zip code 11207 had the highest number of collisions.
- The most common contributing factors for crashes were driver distraction, unsafe driving, and vehicle failure.
- Car (sedan or SUV) and van were the most common vehicle types involved in crashes.
- The training data showed 967 fatalities.
- Data indicated a relationship between the number of fatalities and the number of injuries in crashes.
- Certain vehicle types were more prone to be involved in fatal crashes.
- There was a large increase in personnel injuries starting in March 2020, along with a significant decrease in vehicle crashes during the same period.

You can explore the interactive dashboard for more insights: [City of New York Motor Vehicle Collisions Dashboard](https://public.tableau.com/views/CityofNewYorkMotorVehicleCollisions/injuryboard?:language=en-US&:display_count=n&:origin=viz_share_link)

![Sample Image](attachment:1425f706-a426-42fd-966b-af9a9435f415.png)

## Conclusion

### Summary

This project provides valuable insights into motor vehicle collisions in the City of New York. It identifies seasonal trends, high-risk areas, and contributing factors, which can be used to enhance road safety measures.

### Next Steps

In future iterations, I plan to:

- Implement machine learning models to predict accident outcomes.
- Explore the effectiveness of safety measures and interventions.
- Continuously update and maintain the dashboard with real-time data.

## Data Dictionary

|Column Name|Description|Data Type|
|----|----|----|
|**CRASH DATE**|Occurrence date of collision|Timestamp|
|**CRASH TIME**|Occurrence time of collision|Text|
|**BOROUGH**|Borough where collision occurred|Text|
|**ZIP CODE**|Postal code of incident occurrence|Text|
|**LATITUDE**|Latitude coordinate for Global Coordinate System, WGS 1984, decimal degrees (EPSG 4326)|Number|
|**LONGITUDE**|Longitude coordinate for Global Coordinate System, WGS 1984, decimal degrees (EPSG 4326)|Number|
|**LOCATION**|Latitude, Longitude pair|Location|
|**ON STREET NAME**|Street on which the collision occurred|Text|
|**CROSS STREET NAME**|Nearest cross street to the collision|Text|
|**OFF STREET NAME**|Street address if known|Text|
|**NUMBER OF PERSONS INJURED**|Number of persons injured|Number|
|**NUMBER OF PERSONS KILLED**|Number of persons killed|Number|
|**NUMBER OF PEDESTRIANS INJURED**| Number of pedestrians injured|Number|
|**NUMBER OF PEDESTRIANS KILLED**|Number of pedestrians killed|Number|
|**NUMBER OF CYCLISTS INJURED**|Number of cyclists injured|Number|
|**NUMBER OF CYCLISTS KILLED**|Number of cyclists killed|Number|
|**NUMBER OF MOTORISTS INJURED**|Number of vehicle occupants injured|Number|
|**NUMBER OF MOTORISTS KILLED**|Number of vehicle occupants killed|Number|
|**CONTRIBUTING FACTOR VEHICLE 1**|Factors contributing to the collision for Vehicle 1|Text|
|**CONTRIBUTING FACTOR VEHICLE 2**|Factors contributing to the collision for Vehicle 2|Text|
|**CONTRIBUTING FACTOR VEHICLE 3**|Factors contributing to the collision for Vehicle 3|Text|
|**CONTRIBUTING FACTOR VEHICLE 4**|Factors contributing to the collision for Vehicle 4|Text|
|**CONTRIBUTING FACTOR VEHICLE 5**|Factors contributing to the collision for Vehicle 5|Text|
|**COLLISION_ID**|Unique record code generated by the system. Primary Key for Crash table|Number|
|**VEHICLE TYPE CODE 1**|Type of vehicle based on the selected vehicle category (ATV, bicycle, car/suv, ebike, escooter, truck/bus, motorcycle, other)|Text|
|**VEHICLE TYPE CODE 2**|Type of vehicle based on the selected vehicle category (ATV, bicycle, car/suv, ebike, escooter, truck/bus, motorcycle, other)|Text|
|**VEHICLE TYPE CODE 3**|Type of vehicle based on the selected vehicle category (ATV, bicycle, car/suv, ebike, escooter, truck/bus, motorcycle, other)|Text|
|**VEHICLE TYPE CODE 4**|Type of vehicle based on the selected vehicle category (ATV, bicycle, car/suv, ebike, escooter, truck/bus, motorcycle, other)|Text|
|**VEHICLE TYPE CODE 5**|Type of vehicle based on the selected vehicle category (ATV, bicycle, car/suv, ebike, escooter, truck/bus, motorcycle, other)|Text|


[Back to top](#top)


