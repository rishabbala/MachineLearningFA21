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

p = 1
kernel_train = (train_data.iloc[:, :-1] @ train_data.iloc[:, :-1].T).pow(p)
kernel_train = kernel_train.multiply(train_data.iloc[:, -1], axis=1)
kernel_dev = (dev_data.iloc[:, :-1] @ train_data.iloc[:, :-1].T).pow(p)
kernel_dev = kernel_dev.multiply(train_data.iloc[:, -1], axis=1)
iter = 0
alpha = np.zeros((train_data.shape[0], 1))
training_accuracy = []
dev_accuracy = []
while iter < max_iter:
    iter += 1
    alpha += np.where(((kernel_train @ alpha).multiply(train_data.iloc[:, -1], axis=0)) <= 0, 1, 0)
    training_accuracy.append(np.sum(np.where((kernel_train @ alpha).multiply(train_data.iloc[:, -1], axis=0) <= 0, 0, 1))/train_data_size)
    dev_accuracy.append(np.sum(np.where((kernel_dev @ alpha).multiply(dev_data.iloc[:, -1], axis=0) <= 0, 0, 1))/dev_data_size)
    print(iter)
alpha = np.sort(alpha)
np.savetxt(f"Part2b_p={p}_alpha.txt", alpha)
np.savetxt(f"Part2b_p={p}_best.txt", [np.amax(training_accuracy), np.argmax(training_accuracy), np.amax(dev_accuracy), np.argmax(dev_accuracy)])
plt.cla()
plt.clf()
plt.plot(np.arange(max_iter)+1, training_accuracy, color="tab:blue")
plt.plot(np.arange(max_iter)+1, dev_accuracy, color="tab:orange")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title(f"Kernalized_batch_perceptron (p={p})")
plt.legend(["Train Accuracy", "Dev Accuracy"])
plt.pause(1)
plt.savefig(f"Kernalized_batch_perceptron (p={p}).png")

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
        alpha += np.where(np.multiply(kernel_dev @ alpha, dev_data.iloc[:n, -1]) <= 0, 1, 0)
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
plt.title(f"Kernalized_perceptron Runtime")
plt.pause(1)
plt.savefig(f"part2b_runtime.png")
