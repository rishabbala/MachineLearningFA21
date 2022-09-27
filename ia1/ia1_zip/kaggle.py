import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import csv

# TRAIN_PATH = 'IA1_train.csv'
TRAIN_PATH = 'PA1_train1.csv'
DEV_PATH = 'IA1_dev.csv'
TEST_PATH = 'PA1_test1.csv'

## Part 0

# w = np.expand_dims(np.array([5.354404, -0.283490, 0.376356, 0.608094, 0.101796, 0.030392, 1.063610, 0.533404, 0.203090, 1.089860, 0.851731, 0.199073, -0.115264, 0.314962, -0.244274, 0.831039, -0.295792, 0.189728, -0.141451, 0.121178, -0.042545, 0.222656, 0.670053]), axis=1)
data = pd.read_csv(TRAIN_PATH)
dev_data = pd.read_csv(DEV_PATH)
test_data = pd.read_csv(TEST_PATH)
id = test_data['id']
data.drop(labels='id', inplace=True, axis=1)
dev_data.drop(labels='id', inplace=True, axis=1)
test_data.drop(labels='id', inplace=True, axis=1)

data[['month', 'day', 'year']] = data.date.str.split(pat='/', expand=True).astype(np.int64)
data.drop(labels='date', inplace=True, axis=1)
dev_data[['month', 'day', 'year']] = dev_data.date.str.split(pat='/', expand=True).astype(np.int64)
dev_data.drop(labels='date', inplace=True, axis=1)
test_data[['month', 'day', 'year']] = test_data.date.str.split(pat='/', expand=True).astype(np.int64)
test_data.drop(labels='date', inplace=True, axis=1)

data['dummy'] = np.array([1.0]*data.shape[0])
dev_data['dummy'] = np.array([1.0]*dev_data.shape[0])
test_data['dummy'] = np.array([1.0]*test_data.shape[0])

data['age_since_renovation'] = np.where(data['yr_renovated'] == 0, data['year'] - data['yr_built'], data['year'] - data['yr_renovated'])
dev_data['age_since_renovation'] = np.where(dev_data['yr_renovated'] == 0, dev_data['year'] - dev_data['yr_built'], dev_data['year'] - dev_data['yr_renovated'])
test_data['age_since_renovation'] = np.where(test_data['yr_renovated'] == 0, test_data['year'] - test_data['yr_built'], test_data['year'] - test_data['yr_renovated'])

## Add new feature sqft_living15/sqft_lot15  Best Features Reduces losses
data['floors*sqft_above'] = data['floors']*data['sqft_above']
dev_data['floors*sqft_above'] = dev_data['floors']*dev_data['sqft_above']
test_data['floors*sqft_above'] = test_data['floors']*test_data['sqft_above']
##

## Add new feature condition*sqft_living  Best Features Reduces losses
data['condition*sqft_living'] = data['condition']*data['sqft_living']
dev_data['condition*sqft_living'] = dev_data['condition']*dev_data['sqft_living']
test_data['condition*sqft_living'] = test_data['condition']*test_data['sqft_living']
##

## Add new feature grade*sqft_living THE Best Features Reduces losses
data['grade*sqft_living'] = data['grade']*data['sqft_living']
dev_data['grade*sqft_living'] = dev_data['grade']*dev_data['sqft_living']
test_data['grade*sqft_living'] = test_data['grade']*test_data['sqft_living']
##

## Add new feature condition*sqft_living THE Best Features Reduces losses
data['condition*sqft_living'] = data['condition']*data['sqft_living']
dev_data['condition*sqft_living'] = dev_data['condition']*dev_data['sqft_living']
test_data['condition*sqft_living'] = test_data['condition']*test_data['sqft_living']
##

# ## Add new feature bed/bath  Best Features reduces both losses
data['bed/bath'] = data['bedrooms']/data['bathrooms']
dev_data['bed/bath'] = dev_data['bedrooms']/dev_data['bathrooms']
test_data['bed/bath'] = test_data['bedrooms']/test_data['bathrooms']
# ####

## Add new feature sqft_living/sqft_lot  Best feature Reduces both losses
data['sqft_living/sqft_lot'] = data['sqft_living']/data['sqft_lot']
dev_data['sqft_living/sqft_lot'] = dev_data['sqft_living']/dev_data['sqft_lot']
test_data['sqft_living/sqft_lot'] = test_data['sqft_living']/test_data['sqft_lot']
# ##

## Add new feature sqft_living/sqft_lot  Best feature Reduces both losses
data['sqft_lot/floors'] = data['sqft_lot']/data['floors']
dev_data['sqft_lot/floors'] = dev_data['sqft_lot']/dev_data['floors']
test_data['sqft_lot/floors'] = test_data['sqft_lot']/test_data['floors']
# ##

## Not so good Feature
# data['no_shower'] = np.where(data['bathrooms']%2 != 0, 1, 0)
# dev_data['no_shower'] = np.where(dev_data['bathrooms']%2 != 0, 1, 0)
# test_data['no_shower'] = np.where(test_data['bathrooms']%2 != 0, 1, 0)
###

# ## Using encoded date instead of date Not so good Features
# data['encoded_date'] = data['year'] + data['month']*100 + data['day']
# data.drop(['month', 'day', 'year'], inplace=True, axis=1)
# dev_data['encoded_date'] = dev_data['year'] + dev_data['month']*100 + dev_data['day']
# dev_data.drop(['month', 'day', 'year'], inplace=True, axis=1)
# test_data['encoded_date'] = test_data['year'] + test_data['month']*100 + test_data['day']
# test_data.drop(['month', 'day', 'year'], inplace=True, axis=1)
# ####

# # ## Add new feature sqft_living/sqft_above  Best feature Reduces both losses
# data['sqft_living/sqft_above'] = data['sqft_living']/data['sqft_above']
# dev_data['sqft_living/sqft_above'] = dev_data['sqft_living']/dev_data['sqft_above']
# test_data['sqft_living/sqft_above'] = test_data['sqft_living']/test_data['sqft_above']
# # ##

# # ## Add new feature sqft_living15/sqft_lot15   Not so Good Feature Reduces both losses
# data['sqft_living15/sqft_lot15'] = data['sqft_living15']/data['sqft_lot15']
# dev_data['sqft_living15/sqft_lot15'] = dev_data['sqft_living15']/dev_data['sqft_lot15']
# test_data['sqft_living15/sqft_lot15'] = test_data['sqft_living15']/test_data['sqft_lot15']
# # ##

# ## Add new feature sqft_living15/sqft_lot15  Best Features Reduces both losses
# data['sqft_living/sqft_living15'] = data['sqft_living']/data['sqft_living15']
# dev_data['sqft_living/sqft_living15'] = dev_data['sqft_living']/dev_data['sqft_living15']
# test_data['sqft_living/sqft_living15'] = test_data['sqft_living']/test_data['sqft_living15']
# ##

## Add new feature bedrooms/floors  Not so good Features Reduces both losses
# data['bedrooms/floors'] = data['bedrooms']/data['floors']
# dev_data['bedrooms/floors'] = dev_data['bedrooms']/dev_data['floors']
# test_data['bedrooms/floors'] = test_data['bedrooms']/test_data['floors']
##

# ## Add new feature sqft_living15/sqft_lot15 ## Not so Good Feature
# data['sqft_lot/sqft_lot15'] = data['sqft_lot']/data['sqft_lot15']
# dev_data['sqft_lot/sqft_lot15'] = dev_data['sqft_lot']/dev_data['sqft_lot15']
# test_data['sqft_lot/sqft_lot15'] = test_data['sqft_lot']/test_data['sqft_lot15']
# ##

# # ## Add new feature sqft_above/sqft_basement ## Bad Feature Goes to nan
# data['sqft_above/sqft_basement'] = data['sqft_above']/data['sqft_basement']
# dev_data['sqft_above/sqft_basement'] = dev_data['sqft_above']/dev_data['sqft_basement']
# test_data['sqft_above/sqft_basement'] = test_data['sqft_above']/test_data['sqft_basement']
# # ##


## Add new feature grade*sqft_living THE Best Features Reduces losses
# data['view/waterfront'] = data['view']/(1+data['waterfront'])
# dev_data['view/waterfront'] = dev_data['view']/(1+dev_data['waterfront'])
# test_data['view/waterfront'] = test_data['view']/(1+test_data['waterfront'])
##

# ### Dropping features with low weights Increases loss slightly 
# 'sqft_lot15', 'sqft_living15', '', 'yr_built', 'yr_renovated', 'lat', 'long', 'month', 'day', 'year'
# data.drop(['age_since_renovation', 'sqft_lot15', 'yr_renovated', 'sqft_lot', 'floors'], axis=1, inplace=True)
# dev_data.drop(['age_since_renovation', 'sqft_lot15', 'yr_renovated', 'sqft_lot', 'floors'], axis=1, inplace=True)
# test_data.drop(['age_since_renovation', 'sqft_lot15', 'yr_renovated', 'sqft_lot', 'floors'], axis=1, inplace=True)
# ###

data = data[["dummy"] + [col for col in data.columns if col != "price" and col != "dummy"] + ["price"]]
dev_data = dev_data[["dummy"] + [col for col in dev_data.columns if col != "price" and col != "dummy"] + ["price"]]
test_data = test_data[["dummy"] + [col for col in test_data.columns if col != "dummy"]]

# for columns in data.columns[:-1]:
# plt.scatter(data['view/waterfront'], data.iloc[:, -1])
# plt.xlabel('bedrooms/floors')
# plt.ylabel("price")
# plt.show()
# exit()

mean_arr = []
sd_arr = []
for columns in data.columns[:-1]:
    if columns not in ["waterfront", "dummy"]:
        mean = np.mean(data[columns])
        sd = np.std(data[columns])
        data[columns] = (data[columns] - mean)/sd
        mean_arr.append(mean)
        sd_arr.append(sd)
    else:
        mean_arr.append(-1)
        sd_arr.append(-1)

c = 0
for columns in test_data.columns:
    if columns not in ["waterfront", "dummy"]:
        test_data[columns] = (test_data[columns] - mean_arr[c])/sd_arr[c]
    c += 1

c = 0
for columns in dev_data.columns[:-1]:
    if columns not in ["waterfront", "dummy"]:
        dev_data[columns] = (dev_data[columns] - mean_arr[c])/sd_arr[c]
    c += 1

w = np.random.rand(data.shape[1]-1, 1)
lr = 0.1
for epoch in range(2000):
    x = data.iloc[:, :-1] ## (8000*23)
    y = np.expand_dims(data.iloc[:, -1], axis=1) ## (8000*1)
    diff = np.dot(x, w) - y
    loss = np.sum(diff**2)/data.shape[0]
    print(loss, epoch, lr)
    grad = 2/data.shape[0] * (np.transpose(data.iloc[:, :-1]) @ diff)
    w = w - lr * grad
    with open ("kaggle.csv", 'w') as file:
        csvwriter = csv.writer(file)
        csvwriter.writerow(["weights", w])
        # csvwriter.writerow(["Learning Rate", lr])
        # csvwriter.writerow(["Loss", loss])


np.expand_dims(w, axis=1)
x = dev_data.iloc[:, :-1] ## (8000*23)
y = np.expand_dims(dev_data.iloc[:, -1], axis=1) ## (8000*1)
diff = np.dot(x, w) - y
loss = np.sum(diff**2)/dev_data.shape[0]
print("Dev Loss : ", loss)


fields = ['id', 'price']
with open("output.csv", 'w') as file:
    csvwriter = csv.writer(file)
    csvwriter.writerow(fields)
    for i in range(test_data.shape[0]):
        y = np.dot(test_data.iloc[i], w)[0]
        csvwriter.writerow([id.iloc[i], y])
# print(data['price'])
