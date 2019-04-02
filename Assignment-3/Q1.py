import pandas as pd
import numpy as np
import nodeclass as nc 
import treeclass as tc 
from sklearn.preprocessing import OneHotEncoder

def getData(filename):
    data = pd.read_csv(filename)
    return data

data = getData('./Data/credit-cards.train.csv')
#data = getData('mytest.csv').values[:,1:]
test_data = getData('./Data/credit-cards.test.csv')
val_data = getData('./Data/credit-cards.val.csv')
print(data.shape)

def preprocess(data):
    #array = data.values[1:,1:].astype(np.float)
    array = data
    cont = [0,4,11,12,13,14,15,16,17,18,19,20,21,22]
    for i in range(len(cont)):
        column = array[:,cont[i]].reshape(-1)
        column_median = np.median(column)
        array[:,cont[i]] = np.where(array[:,cont[i]]>column_median,1,0)
    return array

def one_hot_encoding(train_data,test_data,val_data):
    print(train_data.shape,test_data.shape,val_data.shape)
    full_data = np.vstack((train_data[:,:-1],test_data[:,:-1],val_data[:,:-1]))
    encoder = OneHotEncoder(categories = 'auto')
    encoder.fit(full_data)
    transformed_data = encoder.transform(full_data).toarray()
    transformed_train = transformed_data[:train_data.shape[0],:]
    transformed_test = transformed_data[train_data.shape[0]:train_data.shape[0]+test_data.shape[0],:]
    transformed_val = transformed_data[train_data.shape[0]+test_data.shape[0]:train_data.shape[0]+test_data.shape[0]+val_data.shape[0],:]
    transformed_train = np.c_[transformed_train,train_data[:,-1]]
    transformed_test = np.c_[transformed_test,test_data[:,-1]]
    transformed_val = np.c_[transformed_val,val_data[:,-1]]
    print(transformed_train.shape,transformed_test.shape,transformed_val.shape)
    return transformed_train,transformed_test,transformed_val

data = data.values[1:,1:].astype(np.float)
test_data = test_data.values[1:,1:].astype(np.float)
val_data = val_data.values[1:,1:].astype(np.float)

array = preprocess(data)
test_array = preprocess(test_data)
val_array = preprocess(val_data)
print(array)

#root = nc.DecisionTreeNode()
#tree = tc.DecisionTree(root)
#tree.train(tree.root,array)
#tree.set_node_count()#tree.root)
#tree.traverse(tree.root)
#acc_history = []
#tree.predict(array)#,acc_history)
#tree.plot_this(acc_history)
#tree = tc.DecisionTree(None)
#tree.play_with_library(array,test_array,val_array)

ohe_train, ohe_test, ohe_val = one_hot_encoding(data,test_data,val_data)
tree = tc.DecisionTree(None)
tree.random_forest(array,test_array,val_array)