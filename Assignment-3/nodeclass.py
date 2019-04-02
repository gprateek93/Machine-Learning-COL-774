import numpy as np
from math import log2

class DecisionTreeNode:
    def __init__(self):
        #print('new node created')
        self.children = {}
        self.label = None
        self.split_attr = None
        self.inf_gain = None
        self.nodes_till_now = 0
        self.median = None
        
    def entropy(self,data):
        total = len(data[:,-1])
        if(total == 0):
            return 0
        positive = len(data[data[:,-1]==1])
        negative = len(data[data[:,-1]==0])
        positive_ratio = positive/total
        negative_ratio = negative/total
        H = 0
        if(positive_ratio != 0 and negative_ratio != 0):
            H = -positive_ratio * log2(positive_ratio) + (-negative_ratio * log2(negative_ratio))
        return H
    
    def info_gain(self,X,data,mode):
        total = data.shape[0]
        if(mode == 'median'):
            med = np.median(data[:,X])
            more_than_median = np.array(data[data[:,X]>med])
            less_than_median = np.array(data[data[:,X]<=med])
            I = self.entropy(data) - (more_than_median.shape[0]/total) * self.entropy(more_than_median) - (less_than_median.shape[0]/total)*self.entropy(less_than_median)
        
        else:
            cond_entropy = 0
            x_values = np.unique(data[:,X])
            for i in x_values:
                i_data = np.array(data[data[:,X] == i])
                cond_entropy += (i_data.shape[0]/total) * self.entropy(i_data)
            I = self.entropy(data) - cond_entropy

        return I
    
    def best_attribute(self,data,mode):
        best_I = -1
        best_attr = -1
        for i in range(data.shape[1]-1):
            info_gain_i = self.info_gain(i,data,mode)
            if info_gain_i > best_I:
                best_I = info_gain_i
                best_attr = i
        self.split_attr = best_attr
        self.inf_gain = best_I
        return best_attr
    
    def assign_label(self,label):
        self.label = label