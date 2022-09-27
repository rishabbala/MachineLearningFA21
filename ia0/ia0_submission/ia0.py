import pandas as pd
from pandas.core.algorithms import unique
import numpy as np
import matplotlib.pyplot as plt

FILE_PATH = 'pa0(train-only).csv'
data = pd.read_csv(FILE_PATH)
print(f"Before ID removal: {data.shape}")
## a. Drop the ID entry
data.drop('id', inplace=True, axis=1)
print(data)
print(f"After ID removal: {data.shape}")
# # b. Split the date
print("-----------------------------------------------------")
data[['month','day', 'year']] = data.date.str.split("/",expand=True)
data.drop('date', inplace=True, axis=1)
print(data)
print("-----------------------------------------------------")
# c. Plot a boxplot of change in price with bedroom/bathroom/floors
unique_bedrooms = np.sort(unique(data['bedrooms']))
unique_bathrooms = np.sort(unique(data['bathrooms']))
unique_floors = np.sort(unique(data['floors']))
y_bed = []
print(f"Unique number of bedrooms: {unique_bedrooms}")
print(f"Unique number of bathrooms: {unique_bathrooms}")
print(f"Unique number of floors: {unique_floors}")
for bed in unique_bedrooms:
    y_bed.append(data[(data['bedrooms'] == bed)]['price'].to_list())
boxplot = plt.boxplot(y_bed, showfliers=False)
plt.xticks(np.arange(1, unique_bedrooms.size+1), unique_bedrooms)
plt.ylabel("Price")
plt.xlabel("Number of Bedrooms")
plt.show()

y_bath = []
for bath in unique_bathrooms:
        y_bath.append(data[(data['bathrooms'] == bath)]['price'].to_list())
boxplot = plt.boxplot(y_bath, showfliers=False)
plt.xticks(np.arange(1, unique_bathrooms.size+1, 2), unique_bathrooms[::2])
plt.ylabel("Price")
plt.xlabel("Number of Bathrooms")
plt.show()

y_floor = []
for floor in unique_floors:
    y_floor.append(data[(data['floors'] == floor)]['price'].to_list())
boxplot = plt.boxplot(y_floor, showfliers=False)
plt.xticks(np.arange(1, unique_floors.size+1), unique_floors)
plt.ylabel("Price")
plt.xlabel("Number of Floors")
plt.show()
print("-----------------------------------------------------")

# d. Compute the covariance and correlation and plot scatter plots
covariance = data[['sqft_living', 'sqft_lot', 'sqft_living15', 'sqft_lot15']].cov()
correlation = data[['sqft_living', 'sqft_lot', 'sqft_living15', 'sqft_lot15']].corr()
print(f"COVARIANCE:\n {covariance}\n")
print(f"CORRELATION:\n {correlation}\n")
data.plot.scatter(x='sqft_living', y='sqft_living15')
data.plot.scatter(x='sqft_lot', y='sqft_lot15')
plt.show()
