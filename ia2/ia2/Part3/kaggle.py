import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv

def sigma(x, w):
    return 1/(1+np.exp(-x @ w))

train_data = pd.read_csv("IA2-train.csv")
dev_data = pd.read_csv("IA2-dev.csv")
test_data = pd.read_csv("IA2-test-small-v2-X.csv")
data = pd.concat([train_data, dev_data])

## Feature engineering mentioned in report. This takes a LOT of time.

# policy_channel = []
# for i in range(data.shape[0]):
#     for j in range(1, 164):
#         try: 
#             if data["Policy_Sales_Channel_"+str(j)].iloc[i] == 1:
#                 policy_channel.append(j)
#                 break
#         except KeyError:
#             pass

# data["Policy_Channel"] = np.true_divide(policy_channel, 163)
# data.drop(["Policy_Sales_Channel_"+str(i) for i in range(1, 164)], axis=1, inplace=True, errors="ignore")

# policy_channel = []
# for i in range(test_data.shape[0]):
#     for j in range(1, 164):
#         try: 
#             if test_data["Policy_Sales_Channel_"+str(j)].iloc[i] == 1:
#                 policy_channel.append(j)
#                 break
#         except KeyError:
#             pass

# test_data["Policy_Channel"] = np.true_divide(policy_channel, 163)
# test_data.drop(["Policy_Sales_Channel_"+str(i) for i in range(1, 164)], axis=1, inplace=True, errors="ignore")

# region_code = []
# for i in range(data.shape[0]):
#     for j in range(53):
#         if data["Region_Code_"+str(j)].iloc[i] == 1:
#             region_code.append(j)
#             break

# data["Region_Code"] = np.true_divide(region_code, 52)
# data.drop(["Region_Code_"+str(i) for i in range(53)], axis=1, inplace=True)

# region_code = []
# for i in range(test_data.shape[0]):
#     for j in range(53):
#         if test_data["Region_Code_"+str(j)].iloc[i] == 1:
#             region_code.append(j)
#             break

# test_data["Region_Code"] = np.true_divide(region_code, 52)
# test_data.drop(["Region_Code_"+str(i) for i in range(53)], axis=1, inplace=True)

# vehicle_age = []
# for i in range(data.shape[0]):
#     for j in range(3):
#         if data["Vehicle_Age_"+str(j)].iloc[i] == 1:
#             vehicle_age.append(j)
#             break

# data["Vehicle_Age"] = np.true_divide(vehicle_age, 2)
# data.drop(["Vehicle_Age_"+str(i) for i in range(3)], axis=1, inplace=True)

# vehicle_age = []
# for i in range(test_data.shape[0]):
#     for j in range(3):
#         if test_data["Vehicle_Age_"+str(j)].iloc[i] == 1:
#             vehicle_age.append(j)
#             break

# test_data["Vehicle_Age"] = np.true_divide(vehicle_age, 2)
# test_data.drop(["Vehicle_Age_"+str(i) for i in range(3)], axis=1, inplace=True)



# train_data = data.iloc[:int(0.8*data.shape[0]), :]
# dev_data = data.iloc[int(0.8*data.shape[0]):, :]

# y = np.expand_dims(train_data["Response"], axis=1)
# train_data.drop("Response", axis=1, inplace=True)
# y_dev = np.expand_dims(dev_data["Response"], axis=1)
# dev_data.drop("Response", axis=1, inplace=True)


y = np.expand_dims(data["Response"], axis=1)
data.drop("Response", axis=1, inplace=True)

for c in ["Age", "Annual_Premium", "Vintage"]:
    mean = np.mean(data[c])
    std = np.std(data[c])
    # train_data[c] = (train_data[c] - mean)/std
    # dev_data[c] = (dev_data[c] - mean)/std
    data[c] = (data[c] - mean)/std
    test_data[c] = (test_data[c] - mean)/std


# w = np.random.rand(train_data.shape[1], 1)
w = np.random.rand(data.shape[1], 1)
# reg_lambda_range = list(10**i for i in np.arange(-4, 3, 0.5))
reg_lambda_range = [0.001]
training_limit = 5000

alpha = 0.75
# N = train_data.shape[0]
# N_dev = dev_data.shape[0]
for reg_lambda in reg_lambda_range:
    # w = np.random.rand(train_data.shape[1], 1)
    w = np.random.rand(data.shape[1], 1)
    for epoch in range(training_limit):
        # sigma_train = sigma(train_data, w)
        # sigma_dev = sigma(dev_data, w)
        sigma_train = sigma(data, w)
        train_accuracy = np.sum(np.where(y == np.where(sigma_train > 0.5, 1, 0), 1, 0))/data.shape[0] * 100
        # dev_accuracy = np.sum(np.where(y_dev == np.where(sigma_dev > 0.5, 1, 0), 1, 0))/N_dev * 100
        # print(reg_lambda, epoch, train_accuracy, dev_accuracy)
        print(reg_lambda, epoch, train_accuracy)
        w = w + alpha/data.shape[0] * np.transpose(np.transpose(y - sigma_train) @ data)
        w[1:] = np.sign(w[1:])*np.where(np.abs(w[1:])-alpha*reg_lambda > 0, np.abs(w[1:])-alpha*reg_lambda, 0)

fields = ['ID', 'Response']
with open ("kaggle.csv", 'w') as file:
    csvwriter = csv.writer(file)
    csvwriter.writerow(fields)
    s = np.where(sigma(test_data, w) > 0.5, 1, 0)
    for i in range(s.shape[0]):
        csvwriter.writerow([i, s[i][0]])