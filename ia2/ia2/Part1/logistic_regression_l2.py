import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv

def sigma(x, w):
    return 1/(1+np.exp(-x @ w))

train_data = pd.read_csv("IA2-train.csv")
dev_data = pd.read_csv("IA2-dev.csv")
y = np.expand_dims(train_data.iloc[:, -1], axis=1)
train_data.drop("Response", axis=1, inplace=True)
y_dev = np.expand_dims(dev_data.iloc[:, -1], axis=1)
dev_data.drop("Response", axis=1, inplace=True)

for c in ["Age", "Annual_Premium", "Vintage"]:
    mean = np.mean(train_data[c])
    std = np.std(train_data[c])
    train_data[c] = (train_data[c] - mean)/std
    dev_data[c] = (dev_data[c] - mean)/std

w = np.random.rand(train_data.shape[1], 1)
reg_lambda_range = list(10**i for i in np.arange(-4, 4, 1).astype(float))
# reg_lambda_range = [0.0001, 0.001, 0.01]
training_limit = 500

alpha = 0.1
N = train_data.shape[0]
N_dev = dev_data.shape[0]
train_vs_reg_accuracy_arr = []
dev_vs_reg_accuracy_arr = []
best_acc = 0
best_lambda = 0
sparsity = []
for reg_lambda in reg_lambda_range:
    w = np.random.rand(train_data.shape[1], 1)
    epoch_arr = []
    train_accuracy_arr = []
    dev_accuracy_arr = []
    for epoch in range(training_limit):
        sigma_train = sigma(train_data, w)
        sigma_dev = sigma(dev_data, w)
        train_accuracy = np.sum(np.where(y == np.where(sigma_train >= 0.5, 1, 0), 1, 0))/N * 100
        dev_accuracy = np.sum(np.where(y_dev == np.where(sigma_dev >= 0.5, 1, 0), 1, 0))/N_dev * 100
        if epoch > 1000 and abs(train_accuracy - train_accuracy_arr[-1]) < 0.001 and abs(dev_accuracy - dev_accuracy_arr[-1]) < 0.001:
            break
        print("Lambda:", reg_lambda, "Epoch:", epoch, "Training Accuracy:", train_accuracy, "Validation Accuracy:", dev_accuracy)
        w = w + alpha/N * np.transpose(np.transpose(y - sigma_train) @ train_data)
        w[1:] = w[1:] - alpha*reg_lambda*w[1:]
        train_accuracy_arr.append(train_accuracy)
        dev_accuracy_arr.append(dev_accuracy)
        epoch_arr.append(epoch)
    if dev_accuracy_arr[-1] > best_acc:
        best_acc = dev_accuracy_arr[-1]
        best_lambda = reg_lambda
    sparsity.append(np.sum(np.where(abs(w) == 0, 1, 0)))
    with open ("Part1.csv", 'a') as file:
        csvwriter = csv.writer(file)
        w = np.abs(w)
        w = w.sort_values(w.columns[0], ascending=False)
        csvwriter.writerow(["weights", w[:5]])
        csvwriter.writerow(["Training Accuracy", train_accuracy_arr[-1]])
        csvwriter.writerow(["Validation Loss", dev_accuracy_arr[-1]])
    plt.plot(epoch_arr, train_accuracy_arr, color="tab:blue")
    plt.plot(epoch_arr, dev_accuracy_arr, color="tab:orange")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"L2 Lambda = {reg_lambda}")
    plt.legend(["Train Accuracy", "Dev Accuracy"])
    plt.pause(1)
    plt.savefig(f"L2 lamda = {reg_lambda}.png")
    train_vs_reg_accuracy_arr.append(train_accuracy_arr[np.argmax(dev_accuracy_arr)])
    dev_vs_reg_accuracy_arr.append(np.amax(dev_accuracy_arr))
    plt.clf()
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.scatter(reg_lambda_range, train_vs_reg_accuracy_arr, color="tab:blue")
ax.scatter(reg_lambda_range, dev_vs_reg_accuracy_arr, color="tab:orange")
ax.set_xscale('log')
plt.title(f"L2 Accuracy vs Lambda")
plt.xlabel("Lambda")
plt.ylabel("Accuracy")
plt.legend(["Train Accuracy", "Dev Accuracy"])
plt.pause(1)
plt.savefig(f"L2 Accuracy vs lambda.png")
print("Best Accuracy :", best_acc, "Best lambda :", best_lambda)
plt.plot(reg_lambda_range, sparsity)
ax.set_xscale('log')
plt.xlabel("Lambda")
plt.ylabel("Number of Sparse weights")
plt.title(f"Sparsity of Weights for L2 Regularization")
plt.pause(1)
plt.savefig(f"L2 sparsity_v2.png")