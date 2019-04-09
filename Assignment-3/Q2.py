import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import nnclass as nn
import matplotlib.pyplot as plt 
import sys

def getData(filename):
    data = pd.read_csv(filename, sep=',', header=None)
    return data


def one_hot_encoding(train_data,test_data,train_filename = './default_train.csv',test_filename= './default_test.csv',mode = 'dont_save'):
    #print(train_data.shape,test_data.shape)
    full_data = np.vstack((train_data[:,:-1],test_data[:,:-1]))
    encoder = OneHotEncoder(categories = 'auto')
    encoder.fit(full_data)
    transformed_data = encoder.transform(full_data).toarray()
    transformed_train = transformed_data[:train_data.shape[0],:]
    transformed_test = transformed_data[train_data.shape[0]:train_data.shape[0]+test_data.shape[0],:]
    transformed_train = np.c_[transformed_train,train_data[:,-1]]
    transformed_test = np.c_[transformed_test,test_data[:,-1]]
    #print(transformed_train.shape,transformed_test.shape)
    if(mode == 'save'):
        np.savetxt(train_filename,transformed_train,delimiter=',')
        np.savetxt(test_filename,transformed_test,delimiter=',')
    return transformed_train,transformed_test

def read_config(config_filename):
    with open(config_filename) as fp:
        data = fp.readlines()
    num_inputs = int(data[0])
    num_outputs = int(data[1])
    batch_size = int(data[2])
    num_layers = int(data[3])
    layers = data[4].split(' ')
    layers = list(map(int,layers))
    activation = data[5].rstrip('\n')
    learning_rate = data[6]
    return num_inputs,num_outputs,batch_size,num_layers,layers,activation,learning_rate

def main(args):
    if len(args) == 4:
        print("here")
        train_file = args[0]
        test_file = args[1]
        ohe_train_file = args[2]
        ohe_test_file = args[3]
        train_data = getData(train_file).values
        test_data = getData(test_file).values
        one_hot_encoding(train_data,test_data,train_filename=ohe_train_file,test_filename=ohe_test_file, mode='save')
    else:
        config_file = args[0]
        ohe_train_file = args[1]
        ohe_test_file = args[2]
        ohe_train = getData(ohe_train_file).values
        ohe_test = getData(ohe_test_file).values
        num_inputs,num_outputs,batch_size,num_layers,layers,activation,learning_rate = read_config(config_file)
        if(learning_rate == 'fixed'):
            my_network = nn.network(num_inputs=num_inputs, layers= layers, batch_size= batch_size, num_outputs= num_outputs,activation=activation)
            my_network.train(ohe_train,eta = 0.1,epochs= 500)
            print("Training completed successfully!!")
            acc, conf = my_network.predict(ohe_test)
            print("Accuracy on test_data:", acc)
            print("Confusion matrix on the test_data", conf)
        elif learning_rate == 'adaptive':
            training_loss_prev = 1
            eta = 0.1
            count = 0
            flag = 0
            while(True):  
                print('working on eta',eta) 
                count+=1 
                my_network = nn.network(num_inputs=85, layers= layers, batch_size= 100, num_outputs= 10,activation='relu')
                my_network.train(ohe_train,eta,epochs= 200)
                metric_train = my_network.predict(ohe_train)
                metric_test = my_network.predict(ohe_test)
                print("Train_accuracy",metric_train[0])
                print("Test_accuracy",metric_test[0])
                print(metric_train[1])
                print(metric_test[1])
                training_loss_curr = my_network.loss(ohe_train)
                print("Training loss:",training_loss_curr)

                if(training_loss_prev - training_loss_curr) < 1e-4:
                    flag+=1
                training_loss_prev = training_loss_curr
                if flag >= 2:
                    eta/=5
                if(count%2==0):
                    if(training_loss_prev - training_loss_curr) >= 1e-4:
                        flag = 0
                if(eta<1e-7):
                    break

if __name__ == "__main__":
    if len(sys.argv) == 4 or len(sys.argv) == 5:
        main(sys.argv[1:])
    else:
        sys.stderr("Invalid command line arguments")

            