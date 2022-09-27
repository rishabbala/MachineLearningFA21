import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import csv

TRAIN_PATH = 'IA1_train.csv'
DEV_PATH = 'IA1_dev.csv'

## Part 0
data = pd.read_csv(TRAIN_PATH)
data.drop(labels='id', inplace=True, axis=1)
data[['month', 'day', 'year']] = data.date.str.split(pat='/', expand=True).astype(np.int64)
data.drop(labels='date', inplace=True, axis=1)
data['dummy'] = np.array([1.0]*data.shape[0])
data['age_since_renovation'] = np.where(data['yr_renovated'] == 0, data['year'] - data['yr_built'], data['year'] - data['yr_renovated'])

dev_data = pd.read_csv(DEV_PATH)
dev_data.drop(labels='id', inplace=True, axis=1)
dev_data[['month', 'day', 'year']] = dev_data.date.str.split(pat='/', expand=True).astype(np.int64)
dev_data.drop(labels='date', inplace=True, axis=1)
dev_data['dummy'] = np.array([1.0]*dev_data.shape[0])
dev_data['age_since_renovation'] = np.where(dev_data['yr_renovated'] == 0, dev_data['year'] - dev_data['yr_built'], dev_data['year'] - dev_data['yr_renovated'])

data = data[["dummy"] + [col for col in data.columns if col != "price" and col != "dummy"] + ["price"]]
dev_data = dev_data[["dummy"] + [col for col in dev_data.columns if col != "price" and col != "dummy"] + ["price"]]

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
for columns in dev_data.columns[:-1]:
    if columns not in ["waterfront", "dummy"]:
        dev_data[columns] = (dev_data[columns] - mean_arr[c])/sd_arr[c]
    c += 1


## Part 1
loss_min = np.inf
lr_set = [10**-8, 10**-7, 10**-6, 10**-5, 10**-4, 10**-3, 10**-2, 10**-1, 1, 10]
w = np.random.rand(data.shape[1]-1, 1)
for lr in lr_set:
    loss_arr = []
    epoch_arr = []
    w = np.random.rand(data.shape[1]-1, 1)
    loss = 0
    for epoch in range(5000):
        x = data.iloc[:, :-1] ## (8000*23)
        y = np.expand_dims(data.iloc[:, -1], axis=1) ## (8000*1)
        diff = np.dot(x, w) - y
        loss = np.sum(diff**2)/data.shape[0]
        print(loss, epoch, lr)
        grad = 2/data.shape[0] * (np.transpose(data.iloc[:, :-1]) @ diff)
        w = w - lr * grad
        if (np.isnan(loss).any() == True):
            break
        loss_arr.append(loss)
        epoch_arr.append(epoch)
    # if loss < loss_min:
    #     loss_min = loss
    #     w_best = w
    #     lr_best = lr
    np.expand_dims(w, axis=1)
    x = dev_data.iloc[:, :-1]
    y = np.expand_dims(dev_data.iloc[:, -1], axis=1)
    diff = np.dot(x, w) - y
    dev_loss = np.sum(diff**2)/dev_data.shape[0]
    with open ("Part1.csv", 'a') as file:
        csvwriter = csv.writer(file)
        csvwriter.writerow(["weights", w])
        csvwriter.writerow(["Learning Rate", lr])
        csvwriter.writerow(["Training Loss", loss])
        csvwriter.writerow(["Validation Loss", dev_loss])
    # plt.plot(epoch_arr, loss_arr)
    # plt.legend([str(lr)])
    # plt.xlabel("Epochs")
    # plt.ylabel("MSE")
    # plt.show()

