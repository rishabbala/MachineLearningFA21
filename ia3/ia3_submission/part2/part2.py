import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

train_data = pd.read_csv("IA2-train.csv")
dev_data = pd.read_csv("IA2-dev.csv")

train_data["Response"] = np.where(train_data["Response"] == 0, -1, 1)
dev_data["Response"] = np.where(dev_data["Response"] == 0, -1, 1)

for c in ["Age", "Annual_Premium", "Vintage"]:
    mean = np.mean(train_data[c])
    std = np.std(train_data[c])
    train_data[c] = (train_data[c] - mean)/std
    dev_data[c] = (dev_data[c] - mean)/std

train_data_size = train_data.shape[0]
dev_data_size = dev_data.shape[0]
max_iter = 100

print("For different values of p")

for p in [1, 2, 3, 4, 5]:
    alpha = np.zeros(train_data.shape[0])
    kernel_train = (train_data.iloc[:, :-1] @ train_data.iloc[:, :-1].T).pow(p)
    kernel_train = kernel_train.multiply(train_data.iloc[:, -1], axis=1)
    kernel_dev = (dev_data.iloc[:, :-1] @ train_data.iloc[:, :-1].T).pow(p)
    kernel_dev = kernel_dev.multiply(train_data.iloc[:, -1], axis=1)
    iter = 0
    training_accuracy = []
    dev_accuracy = []
    while iter < max_iter:
        for i in range(train_data_size):
            u_train = kernel_train.iloc[i] @ alpha
            if u_train*train_data.iloc[i, -1] <= 0:
                alpha[i] += 1
        iter += 1
        training_accuracy.append(np.sum(np.where(np.squeeze(np.where(kernel_train @ np.expand_dims(alpha, 1)>=0, 1, -1), 1) == train_data.iloc[:, -1], 1, 0))/train_data_size)
        dev_accuracy.append(np.sum(np.where(np.squeeze(np.where(kernel_dev @ np.expand_dims(alpha, 1)>=0, 1, -1), 1) == dev_data.iloc[:, -1], 1, 0))/dev_data_size)
        print(p, iter)
    print(alpha)
    alpha = np.sort(alpha)
    np.savetxt(f"Part2_p={p}_alpha.txt", alpha)
    np.savetxt(f"Part2_p={p}_best.txt", [np.amax(training_accuracy), np.argmax(training_accuracy), np.amax(dev_accuracy), np.argmax(dev_accuracy)])
    plt.cla()
    plt.clf()
    plt.plot(np.arange(max_iter)+1, training_accuracy, color="tab:blue")
    plt.plot(np.arange(max_iter)+1, dev_accuracy, color="tab:orange")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"Kernalized_perceptron (p={p})")
    plt.legend(["Train Accuracy", "Dev Accuracy"])
    plt.pause(1)
    plt.savefig(f"Kernalized_perceptron (p={p}).png")

print("For Runtime")

p = 1
time_arr = []
for n in [10, 100, 1000, 10000]:
    iter = 0
    alpha = np.zeros(n)
    start_time = time.time()
    kernel_dev = (dev_data.iloc[:n, :-1] @ dev_data.iloc[:n, :-1].T).pow(p)
    kernel_dev = kernel_dev.multiply(dev_data.iloc[:n, -1], axis=1)
    while iter < max_iter:
        for i in range(n):
            u_dev = np.dot(kernel_dev[i], alpha)
            if u_dev*dev_data.iloc[i, -1] <= 0:
                alpha[i] += 1
        iter += 1
        print(iter)
    end_time = time.time()
    time_arr.append(end_time-start_time)
plt.cla()
plt.clf()
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.set_xscale('log')
plt.plot([10, 100, 1000, 10000], time_arr)
plt.xlabel("Number of training examples")
plt.ylabel("Runtime")
plt.legend(["Running_time", "O(nd(n+m))", "O(n^3)"])
plt.title(f"Kernalized_perceptron Runtime")
plt.pause(1)
plt.savefig(f"part2a_runtime.png")
