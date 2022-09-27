from os import replace
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import random
import csv


train_data = pd.read_csv("mushroom-train.csv")
valid_data = pd.read_csv("mushroom-val.csv")
test_data = pd.read_csv("mushroom-test-X.csv")

class Node():
    def __init__(self, dmax):
        '''
        Initialize the Node class
        Defines:
            self.feature -> to hold the features to build the tree upto current node
            self.left -> left child of node
            self.right -> right child of node
            self.data -> indices in the dataset that satisfy conditions upto this point
            self.positive -> indices amongst self.data that are class = 1
            self.negative -> indices amongst self.data that are class = 0
            self.dmax -> max depth of tree from this node
        '''
        self.feature = []
        self.left = False
        self.right = False
        self.leaf = True
        self.data  = []
        self.positive = []
        self.negative = []
        self.dmax = dmax
        self.dt_class = -1

    def SelectBestFeature(self):
        '''
        Find the Best feature that gives maximal information gain
        Defines: 
            left/right -> Set of points from self.data that satisfy/dont satisfy new constraint
            ll/lr -> length of positive/negative list
            l_class0/1 -> number of elements that belong to class 0/1, which satisfy the new constraint
            r_class0/1 -> number of elements that belong to class 0/1, which dont satisfy the new constraint
            cond_entropy1 -> entropy of positive class
            cond_entropy2 -> entropy of negative class
        '''
        information_gain_max = -float("inf")
        feature = "0"
        ret_left = []
        ret_right = []
        p1 = self.positive/(self.positive+self.negative)
        p0 = self.negative/(self.positive+self.negative)
        if p0 == 0:
            H = 0
        else:
            H = -p0*np.log2(p0)
        if p1 == 0:
            H += 0
        else:
            H += -p1*np.log2(p1)
        for f in self.data.columns[:-1]:
            if f in self.feature or f == "class":
                continue
            left = self.data.loc[self.data[f] == 1]
            right = self.data.loc[self.data[f] == 0]
            # right = list(set(self.data).difference(set(left)))
            ll = left.shape[0]
            lr = right.shape[0]
            l_class0 = left.loc[left["class"] == 0].shape[0]
            l_class1 = left.loc[left["class"] == 1].shape[0]
            r_class0 = right.loc[right["class"] == 0].shape[0]
            r_class1 = right.loc[right["class"] == 1].shape[0]
            if (ll == 0):
                cond_entropy1 = 0
            else:
                p0 = l_class0/(l_class0+l_class1)
                p1 = l_class1/(l_class0+l_class1)
                if p0 == 0:
                    cond_entropy1 = 0
                else:
                    cond_entropy1 = -p0*np.log2(p0)
                if p1 == 0:
                    cond_entropy1 += 0
                else:
                    cond_entropy1 += -p1*np.log2(p1)
                cond_entropy1 *= ll/(ll+lr)
            if (lr == 0):
                cond_entropy2 = 0
            else:
                p0 = r_class0/(r_class0+r_class1)
                p1 = r_class1/(r_class0+r_class1)
                if p0 == 0:
                    cond_entropy2 = 0
                else:
                    cond_entropy2 = -p0*np.log2(p0)
                if p1 == 0:
                    cond_entropy2 += 0
                else:
                    cond_entropy2 += -p1*np.log2(p1)
                cond_entropy2 *= lr/(ll+lr)
            inf_gain = H-(cond_entropy1 + cond_entropy2)
            if (inf_gain > information_gain_max):
                information_gain_max = inf_gain
                ret_left = left.copy(deep=True)
                ret_right = right.copy(deep=True)
                feature = f
        print(feature, information_gain_max)
        self.data.drop(feature, inplace=True, axis=1)
        ret_left.drop(feature, inplace=True, axis=1)
        ret_right.drop(feature, inplace=True, axis=1)
        return feature, ret_left, ret_right

    def BuildTree(self):
        '''
        Build the Tree
        Defines:
            Set positive and negative to those indices where the result is 1/0
            Check if the size of positive/negative elements in the node is 0
            If not get the best feature, split the tree into two (left = positive, right = negative)
            Recursively call again
        '''
        self.positive = self.data.loc[self.data["class"] == 1].shape[0]
        self.negative = self.data.loc[self.data["class"] == 0].shape[0]
        if self.positive > self.negative:
            self.dt_class = 1
        else:
            self.dt_class = 0
        if (self.positive == 0 and self.negative == self.data.shape[0]) or (self.positive == self.data.shape[0] and self.negative == 0) or self.dmax == 0:
            self.leaf = True
            print("1.", self.positive, self.negative, self.feature, self.leaf)
            return
        feature, left, right = self.SelectBestFeature()
        self.feature.append(feature)
        self.dmax -= 1
        self.leaf = False
        Left_Node = Node(self.dmax)
        Left_Node.data = left
        Left_Node.feature = list(self.feature)
        Right_Node = Node(self.dmax)
        Right_Node.data = right
        Right_Node.feature = list(self.feature)
        self.left = Left_Node
        self.right = Right_Node
        Left_Node.BuildTree()
        Right_Node.BuildTree()
        print("2.", self.positive, self.negative, self.feature, self.leaf)
        return
    
    def Evaluate(self, valid_data):
        if self.leaf == True:
            return self.dt_class
        elif valid_data[self.feature[-1]] == 1:
            return self.left.Evaluate(valid_data)
        else:
            return self.right.Evaluate(valid_data)

train_accuracy = []
valid_accuracy = []
for dmax in range(1, 11):
    N = Node(dmax)
    N.data = train_data.copy(deep=True)
    N.BuildTree()
    correct = 0
    for i in range(train_data.shape[0]):
        output = N.Evaluate(train_data.iloc[i])
        if output == train_data.iloc[i]["class"]:
            correct += 1
    train_accuracy.append(correct/train_data.shape[0])
    correct = 0
    for i in range(valid_data.shape[0]):
        output = N.Evaluate(valid_data.iloc[i])
        if output == valid_data.iloc[i]["class"]:
            correct += 1
    valid_accuracy.append(correct/valid_data.shape[0])
    print(dmax, train_accuracy[-1], valid_accuracy[-1])

plt.scatter(np.arange(1, 11), train_accuracy, color="tab:blue")
plt.plot(np.arange(1, 11), train_accuracy, color="tab:blue")
plt.scatter(np.arange(1, 11), valid_accuracy, color="tab:orange")
plt.plot(np.arange(1, 11), valid_accuracy, color="tab:orange")
plt.xlabel("dmax")
plt.ylabel("Accuracy")
plt.title(f"Decision Tree Accuracies for different dmax")
plt.legend(["Train Accuracy", "Dev Accuracy"])
plt.pause(1)
plt.savefig(f"DT.png")

# # fields = ['ID', 'Class']
# # with open ("kaggle.csv", 'w') as file:
# #     csvwriter = csv.writer(file)
# #     csvwriter.writerow(fields)
# #     for i in range(test_data.shape[0]):
# #         output = N.Evaluate(test_data.iloc[i])
# #         csvwriter.writerow([i, output])