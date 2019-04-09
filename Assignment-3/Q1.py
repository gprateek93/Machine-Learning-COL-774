import pandas as pd
import numpy as np
import nodeclass as nc
import treeclass as tc 
from sklearn.preprocessing import OneHotEncoder
import sys

def getData(filename):
    data = pd.read_csv(filename)
    return data

def preprocess(data):
    array = data
    cont = [0,4,11,12,13,14,15,16,17,18,19,20,21,22]
    for i in range(len(cont)):
        column = array[:,cont[i]].reshape(-1)
        column_median = np.median(column)
        array[:,cont[i]] = np.where(array[:,cont[i]]>column_median,1,0)
    return array

def one_hot_encoding(train_data,test_data,val_data):
    print(train_data.shape,test_data.shape,val_data.shape)
    cat_data = [2,3,5,6,7,8,9,10]
    full_data = np.vstack((train_data,test_data,val_data))
    use_data = np.vstack((train_data[:,cat_data],test_data[:,cat_data],val_data[:,cat_data]))
    full_data = np.delete(full_data,2,1)
    full_data = np.delete(full_data,2,1)
    full_data = np.delete(full_data,np.s_[3:9],1)
    encoder = OneHotEncoder(categories = 'auto')
    encoder.fit(use_data)
    transformed_data = encoder.transform(use_data).toarray()
    transformed_data = np.c_[transformed_data,full_data]
    transformed_train = transformed_data[:train_data.shape[0],:]
    transformed_test = transformed_data[train_data.shape[0]:train_data.shape[0]+test_data.shape[0],:]
    transformed_val = transformed_data[train_data.shape[0]+test_data.shape[0]:train_data.shape[0]+test_data.shape[0]+val_data.shape[0],:]
    print(transformed_train.shape,transformed_test.shape,transformed_val.shape)
    return transformed_train,transformed_test,transformed_val

def main(args):
    print("here")
    mode = int(args[0])
    train_file = args[1]
    test_file = args[2]
    val_file = args[3]
    data = getData(train_file).values[1:,1:].astype(np.float)
    test_data = getData(test_file).values[1:,1:].astype(np.float)
    val_data = getData(val_file).values[1:,1:].astype(np.float)
    if mode == 1:
        array = preprocess(data)
        test_array = preprocess(test_data)
        val_array = preprocess(val_data)
        root = nc.DecisionTreeNode()
        tree = tc.DecisionTree(root)
        tree.train(tree.root,array,array,test_array,val_array)
        tree.predict(array)
        tree.predict(test_array)
        tree.predict(val_array)
        #tree.plot_this()
    elif mode == 2:
        array = preprocess(data)
        test_array = preprocess(test_data)
        val_array = preprocess(val_data)
        root = nc.DecisionTreeNode()
        tree = tc.DecisionTree(root)
        tree.train(tree.root,array,array,test_array,val_array)
        tree.post_pruning(array,test_array,val_array)
        tree.predict(array)
        tree.predict(test_array)
        tree.predict(val_array)
        #tree.plot_this()
    elif mode == 3:
        root = nc.DecisionTreeNode()
        tree = tc.DecisionTree(root)
        tree.train(tree.root,data,data,test_data,val_data,mode = 'median')
        tree.predict(data,'median')
        tree.predict(test_data,'median')
        tree.predict(val_data,'median')
        tree.plot_this()
    elif mode == 4:
        array = preprocess(data)
        test_array = preprocess(test_data)
        val_array = preprocess(val_data)
        tree = tc.DecisionTree(None)
        tree.play_with_library(array,test_array,val_array)
    elif mode == 5:
        tree = tc.DecisionTree(None)
        ohe_train, ohe_test, ohe_val = one_hot_encoding(data,test_data,val_data)
        tree.play_with_library(ohe_train,ohe_test,ohe_val)
    elif mode == 6:
        array = preprocess(data)
        test_array = preprocess(test_data)
        val_array = preprocess(val_data)
        tree = tc.DecisionTree(None)
        tree.random_forest(array,test_array,val_array)
        
if __name__ == "__main__":
    print(len(sys.argv))
    if len(sys.argv) != 5:
        sys.stderr("Invalid command line arguments")
    else:
        main(sys.argv[1:])