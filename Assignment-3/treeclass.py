import nodeclass as nc
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.metrics import accuracy_score

class DecisionTree:
    def __init__(self,node):
        #print('Decision Tree with a root created')
        self.root = node
    
    def train(self,root,data):
        #print(data.shape)
        if len(data[data[:,-1] == 1]) == len(data[:,-1]):
            root.assign_label(1)
            return 
        elif len(data[data[:,-1] == 0]) == len(data[:,-1]):
            root.assign_label(0)
            return
        elif data.shape[1] == 1:
            if len(data[data[:,-1] == 0]) > len(data[data[:,-1] == 1]):
                root.assign_label(0)
            else:
                root.assign_label(1)
            return 

        else:
            if len(data[data[:,-1] == 0]) > len(data[data[:,-1] == 1]):
                root.assign_label(0)
            else:
                root.assign_label(1)
            split_attribute = root.best_attribute(data)
            values = np.unique(data[:,split_attribute])
            for val in values:
                val_data = np.array(data[data[:,split_attribute] == val])
                if(root.inf_gain < 1e-10):
                    continue
                #np.delete(val_data,split_attribute,1)
                child = nc.DecisionTreeNode()
                root.children[val] = child
                self.train(child,val_data)
            return

    def predictUtil(self,root,data,y_pred):
        if root:
            y_pred.append(root.label)
            if root.children:
                val = data[root.split_attr]
                if(val in root.children):
                    self.predictUtil(root.children[val],data,y_pred)

    
    def predict(self,data):
        num_of_nodes_max = self.height(self.root)
        print(num_of_nodes_max)
        acc = np.empty((1,3))
        yfinal = []
        for row in data:
            y_pred = []
            self.predictUtil(self.root,row,y_pred)
            yfinal.append(y_pred[-1])
            y_true = row[-1]
            #print(y_pred)
            # y_true = np.array(y_true)
            # y_pred = np.array(y_pred)
            # row_acc = np.where(y_pred == y_true,1,0)
            # np.append(acc,row_acc.reshape(1,-1),axis = 0)
        #     print(acc.shape)
        # print(acc.shape)
        # #acc = np.array(acc)
        # total = len(acc)
        
        '''print(total,num_of_nodes_max)
        acc = np.sum(acc,0) / total
        nodes = range(1,num_of_nodes+1)
        plt.plot(nodes,acc)'''
        acc = accuracy_score(data[:,-1],yfinal)
        print(acc*100)

    def height(self,root):
        if(root):
            if root.children:
                m = 0
                for child in root.children.values():
                    h = self.height(child)
                    if h>m:
                        m = h
                return m+1
            else:
                return 1
    def traverse(self,root):
        if(root):
            if root.children:
                for child in root.children:
                    self.traverse(root.children[child])
            #else:
                #print(root.label)