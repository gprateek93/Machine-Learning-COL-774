import pandas as pd
import numpy as np
import nodeclass as nc 
import treeclass as tc 

def getData(filename):
    data = pd.read_csv(filename)
    return data

data = getData('./Data/credit-cards.train.csv')
#data = getData('mytest.csv').values[:,1:]
test_data = getData('./Data/credit-cards.test.csv')
val_data = getData('./Data/credit-cards.val.csv')
print(data.shape)

def preprocess(data):
    array = data.values[1:,1:].astype(np.float)
    cont = [0,4,11,12,13,14,15,16,17,18,19,20,21,22]
    for i in range(len(cont)):
        column = array[:,cont[i]].reshape(-1)
        column_median = np.median(column)
        array[:,cont[i]] = np.where(array[:,cont[i]]>column_median,1,0)
    return array

array = preprocess(data)
test_array = preprocess(test_data)
val_array = preprocess(val_data)
print(array)

root = nc.DecisionTreeNode()
tree = tc.DecisionTree(root)
tree.train(tree.root,array)
tree.traverse(tree.root)
tree.predict(array)