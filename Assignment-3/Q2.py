import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import nnclass as nn

def getData(filename):
    data = pd.read_csv(filename, sep=',', header=None)
    return data

train_data = getData('./Data/poker-hand-training-true.data.txt').values
test_data = getData('./Data/poker-hand-testing.data.txt').values

def one_hot_encoding(train_data,test_data):
    print(train_data.shape,test_data.shape)
    full_data = np.vstack((train_data[:,:-1],test_data[:,:-1]))
    encoder = OneHotEncoder(categories = 'auto')
    encoder.fit(full_data)
    transformed_data = encoder.transform(full_data).toarray()
    transformed_train = transformed_data[:train_data.shape[0],:]
    transformed_test = transformed_data[train_data.shape[0]:train_data.shape[0]+test_data.shape[0],:]
    transformed_train = np.c_[transformed_train,train_data[:,-1]]
    transformed_test = np.c_[transformed_test,test_data[:,-1]]
    print(transformed_train.shape,transformed_test.shape)
    return transformed_train,transformed_test

ohe_train,ohe_test = one_hot_encoding(train_data,test_data)
my_network = nn.network(num_inputs=85, layers= [15,25], batch_size= 10, num_outputs= 10)
my_network.train(ohe_train,0.1,epochs= 20)
print("Training completed successfully!!")
my_network.predict(ohe_train)