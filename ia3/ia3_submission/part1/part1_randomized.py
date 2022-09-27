import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

train_data = pd.read_csv("IA2-train.csv")
dev_data = pd.read_csv("IA2-dev.csv")

train_data["Response"] = np.where(train_data["Response"] == 0, -1, 1)
dev_data["Response"] = np.where(dev_data["Response"] == 0, -1, 1)

for c in ["Age", "Annual_Premium", "Vintage"]:
    mean = np.mean(train_data[c])
    std = np.std(train_data[c])
    train_data[c] = (train_data[c] - mean)/std
    dev_data[c] = (dev_data[c] - mean)/std

w = np.zeros((train_data.shape[1]-1, 1))
w_hat = np.zeros((train_data.shape[1]-1, 1))
s = 1
train_data_size = train_data.shape[0]
dev_data_size = dev_data.shape[0]

max_iter = 100
iter = 0
training_accuracy_online = []
dev_accuracy_online = []
training_accuracy_avg = []
dev_accuracy_avg = []
while iter < max_iter:
    train_data = train_data.iloc[np.random.permutation(len(train_data))]
    dev_data = dev_data.iloc[np.random.permutation(len(dev_data))]
    for i in range(train_data_size):
        if train_data.iloc[i, -1]*(train_data.iloc[i, :-1] @ w) <= 0:
            w = w + np.multiply(train_data.iloc[i, -1], np.expand_dims(train_data.iloc[i, :-1], 1))
        w_hat = (np.multiply(w_hat, s) + w)/(s+1)
        s = s+1
    iter += 1
    training_accuracy_online.append(np.sum(np.where((train_data.iloc[:, :-1] @ w).multiply(train_data.iloc[:, -1], axis=0)>=0, 1, 0))/train_data_size)
    dev_accuracy_online.append(np.sum(np.where((dev_data.iloc[:, :-1] @ w).multiply(dev_data.iloc[:, -1], axis=0)>=0, 1, 0))/dev_data_size)
    training_accuracy_avg.append(np.sum(np.where((train_data.iloc[:, :-1] @ w_hat).multiply(train_data.iloc[:, -1], axis=0)>=0, 1, 0))/train_data_size)
    dev_accuracy_avg.append(np.sum(np.where((dev_data.iloc[:, :-1] @ w_hat).multiply(dev_data.iloc[:, -1], axis=0)>=0, 1, 0))/dev_data_size)
    print(iter, dev_accuracy_online[-1], dev_accuracy_avg[-1])
print(w)
print(w_hat)
w = np.sort(w)
w_hat = np.sort(w_hat)
np.savetxt(f"Part1_random_w,w_hat.txt", [w.squeeze(1), w_hat.squeeze(1)])
np.savetxt(f"Part1_random_best.txt", [np.amax(training_accuracy_online), np.argmax(training_accuracy_online), np.amax(dev_accuracy_online), np.argmax(dev_accuracy_online), np.amax(training_accuracy_avg), np.argmax(training_accuracy_avg), np.amax(dev_accuracy_avg), np.argmax(dev_accuracy_avg)])
plt.plot(np.arange(max_iter)+1, training_accuracy_online, color="tab:blue")
plt.plot(np.arange(max_iter)+1, dev_accuracy_online, color="tab:orange")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title(f"Online_perceptron + Shuffled Data")
plt.legend(["Train Accuracy", "Dev Accuracy"])
plt.pause(1)
plt.savefig(f"Online_perceptron_Random.png")
plt.cla()
plt.clf()
plt.plot(np.arange(max_iter)+1, training_accuracy_avg, color="tab:blue")
plt.plot(np.arange(max_iter)+1, dev_accuracy_avg, color="tab:orange")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title(f"Average_perceptron + Shuffled Data")
plt.legend(["Train Accuracy", "Dev Accuracy"])
plt.pause(1)
plt.savefig(f"Average_perceptron_Random.png")
plt.cla()
plt.clf()
plt.plot(np.arange(max_iter)+1, training_accuracy_online)
plt.plot(np.arange(max_iter)+1, dev_accuracy_online)
plt.plot(np.arange(max_iter)+1, training_accuracy_avg)
plt.plot(np.arange(max_iter)+1, dev_accuracy_avg)
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title(f"Perceptron Accuracies (with data shuffling)")
plt.legend(["Train Accuracy for Online Perceptron", "Dev Accuracy for Online Perceptron", "Training Accuracy for Avg Perceptron", "Dev Accuracy for for Avg Perceptron"])
plt.pause(1)
plt.savefig(f"Perceptron_Random.png")