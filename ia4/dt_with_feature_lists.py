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
    def __init__(self):
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
        self.dmax = 10
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
        information_gain_max = 0#-float("inf")
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
        for f in train_data.columns[:-1]:
            if f in self.feature or f == "class":
                continue
            left = self.data.loc[self.data[f] == 1]
            right = self.data.loc[self.data[f] == 0]
            # right = list(set(self.data).difference(set(left)))
            # left = np.where(train_data.iloc[self.data][f] == 1)[0]
            # right = np.where(train_data.iloc[self.data][f] == 0)[0]
            ll = left.shape[0]
            lr = right.shape[0]
            # print(ll+lr-len(self.data))
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
            # print(inf_gain, f, cond_entropy1, cond_entropy2)
            # print(H, cond_entropy1, cond_entropy2)
            # if inf_gain<0:
            #     print(f, H, cond_entropy1, cond_entropy2, ll+lr-len(self.data), len(list(set(self.data).difference(set(left)).difference(right))))
            if (inf_gain > information_gain_max):
                information_gain_max = inf_gain
                ret_left = left.copy(deep=True)
                ret_right = right.copy(deep=True)
                feature = f
        # exit()
        # if feature == "0":
        #     print("Yes")
        #     print(feature, self.feature)
        #     exit()
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
        Left_Node = Node()
        Left_Node.feature = list(self.feature)
        Left_Node.data = left
        Left_Node.dmax = self.dmax
        Right_Node = Node()
        Right_Node.feature = list(self.feature)
        Right_Node.data = right
        Right_Node.dmax = self.dmax
        self.left = Left_Node
        self.right = Right_Node
        Left_Node.BuildTree()
        Right_Node.BuildTree()
        print("2.", self.positive, self.negative, self.feature, self.leaf)
        return
    
    def Evaluate(self, valid_data):
        # print(self.feature[-1])
        if self.leaf == True:
            # lp = len(self.positive)
            # ln = len(self.negative)
            # prob_positive = lp/(lp+ln)
            # if random.randint(0, 100)<=prob_positive*100:
            # # if lp>ln:
            #     r = 1
            # else:
            #     r = 0
            # return r
            return self.dt_class
        elif valid_data[self.feature[-1]] == 1:
            return self.left.Evaluate(valid_data)
        else:
            return self.right.Evaluate(valid_data)

N = Node()
N.data = train_data.copy(deep=True)
N.BuildTree()
correct = 0
for i in range(valid_data.shape[0]):
    output = N.Evaluate(valid_data.iloc[i])
    if output == valid_data.iloc[i]["class"]:
        correct += 1
print(correct/valid_data.shape[0])

fields = ['ID', 'Class']
with open ("kaggle.csv", 'w') as file:
    csvwriter = csv.writer(file)
    csvwriter.writerow(fields)
    for i in range(test_data.shape[0]):
        output = N.Evaluate(test_data.iloc[i])
        csvwriter.writerow([i, output])