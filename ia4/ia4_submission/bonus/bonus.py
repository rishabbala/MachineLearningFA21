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
        for f in self.data.columns[:-2]:
            if f in self.feature or f == "class" or f == "weight":
                continue
            left = self.data.loc[self.data[f] == 1]
            right = self.data.loc[self.data[f] == 0]
            feature_left_weight = left["weight"]
            feature_right_weight = right["weight"]
            ll = np.sum(feature_left_weight)
            lr = np.sum(feature_right_weight)
            l_class0 = np.sum(np.where(left["class"] == 0, feature_left_weight, 0))
            l_class1 = np.sum(np.where(left["class"] == 1, feature_left_weight, 0))
            r_class0 = np.sum(np.where(right["class"] == 0, feature_right_weight, 0))
            r_class1 = np.sum(np.where(right["class"] == 1, feature_right_weight, 0))
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

        self.positive = np.sum(np.where(self.data["class"] == 1, self.data["weight"], 0))
        self.negative = np.sum(np.where(self.data["class"] == 0, self.data["weight"], 0))
        if self.positive > self.negative:
            self.dt_class = 1
        else:
            self.dt_class = 0
        if self.positive == 0 or self.negative == 0 or self.dmax == 0:
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

D = [1, 2, 5]
T = [10, 20, 30, 40, 50]
t = 50
for dmax in D:
    train_data["weight"] = np.array([1/train_data.shape[0]]*train_data.shape[0])
    cols = train_data.columns.tolist()
    cols = cols[:-2] + [cols[-1]] + [cols[-2]]
    train_data = train_data[cols]
    boosted_trees = [Node(dmax) for _ in range(t)]
    alpha_arr = []
    for i in range(t):
        N = boosted_trees[i]
        N.data = train_data.copy(deep=True)
        N.BuildTree()
        error = 0
        output_arr = []
        for i in range(train_data.shape[0]):
            output = N.Evaluate(train_data.iloc[i])
            output_arr.append(output)
            if output != train_data.iloc[i]["class"]:
                error += train_data.iloc[i]["weight"]
        if error == 0:
            alpha = 0
        else:
            alpha = 0.5*np.log((1-error)/error)
        alpha_arr.append(alpha)
        train_data["weight"] = np.where(train_data["class"] == output_arr, train_data["weight"]*(np.exp(-alpha)), train_data["weight"]*(np.exp(alpha)))
        train_data["weight"] /= np.sum(train_data["weight"])
    training_accuracy_arr = []
    validation_accuracy_arr = []
    for siz in T:
        correct = 0
        for i in range(train_data.shape[0]):
            o = 0
            z = 0
            j = 0
            for N in boosted_trees[:siz]:
                output = N.Evaluate(train_data.iloc[i])
                if output == 1:
                    o += alpha_arr[j]
                else:
                    z += alpha_arr[j]
                j += 1
            if o > z:
                output = 1
            else:
                output = 0
            if output == train_data.iloc[i]["class"]:
                correct += 1
        training_accuracy_arr.append(correct/train_data.shape[0])
        correct = 0
        for i in range(valid_data.shape[0]):
            o = 0
            z = 0
            j = 0
            for N in boosted_trees[:siz]:
                output = N.Evaluate(valid_data.iloc[i])
                if output == 1:
                    o += alpha_arr[j]
                else:
                    z += alpha_arr[j]
                j += 1
            if o > z:
                output = 1
            else:
                output = 0
            if output == valid_data.iloc[i]["class"]:
                correct += 1
        validation_accuracy_arr.append(correct/valid_data.shape[0])
    plt.scatter(T, training_accuracy_arr)
    plt.plot(T, training_accuracy_arr)
    plt.scatter(T, validation_accuracy_arr)
    plt.plot(T, validation_accuracy_arr)
    plt.ylabel("Accuracy")
    plt.xlabel("Number of trees")
    plt.title(f"Adaboosted Trees Accuracies (dmax = {dmax})")
    plt.legend([f"Training Accuracy", "Validation Accuracy"])
    plt.pause(1)
    plt.savefig(f"Adaboost_dmax={dmax}.png")
    plt.cla()
    plt.clf()
        # print("Positive Weight", np.sum(np.where(train_data["class"] == output_arr, train_data["weight"], 0)))
        # print("Negative Weight", np.sum(np.where(train_data["class"] != output_arr, train_data["weight"], 0)))

# # fields = ['ID', 'Class']
# # with open ("kaggle.csv", 'w') as file:
# #     csvwriter = csv.writer(file)
# #     csvwriter.writerow(fields)
# #     for i in range(test_data.shape[0]):
# #         output = N.Evaluate(test_data.iloc[i])
# #         csvwriter.writerow([i, output])