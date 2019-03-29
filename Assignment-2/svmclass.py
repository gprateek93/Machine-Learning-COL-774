#!/usr/bin/env python
# coding: utf-8

# In[9]:


from cvxopt import matrix, solvers
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics.pairwise import euclidean_distances
from itertools import combinations
from collections import Counter
import pickle
import os
from svmutil import * #from libsvm/python package
from svm import * #from libsvm/python package
file1 = 'train.csv'
file2 = 'test.csv'
data1 = pd.read_csv(file1, header = None)
data2 = pd.read_csv(file2, header = None)


# In[10]:


def preprocess_data(data = data1,mode = 'b',d1 = 5, d2 = -1):
    if(mode == 'm'):
        features = data.iloc[:,:-1]
        features = features.values
        features = features/255
        labels = data.iloc[:,-1]
        labels = labels.values
        labels = labels.astype(float)
    else:
        if d2 == -1:
            d2 = (d1+1)%10
        headings = list(data)
        subset_data_d1 = data[headings[-1]] == d1
        subset_data_d2 = data[headings[-1]] == d2
        subset_data = data[subset_data_d1 | subset_data_d2]
        features = subset_data.iloc[:,:-1]
        labels = subset_data.iloc[:,-1]
        features = features.values
        labels = labels.values
        labels = np.where(labels == d1,1,-1)
        labels = labels.astype(float)
        features = features / 255
    return features,labels


# In[3]:


class SVM_binaryclass:
    def __init__(self,kernel,gamma = 1):
        self.kernel = kernel
        self.gamma = gamma

    def get_parameters(self,features,labels):
        size = features.shape[0]
        y = labels
        x = features
        if self.kernel == 'g':
            phi = euclidean_distances(x,x,squared=True)
            phi = np.exp(-self.gamma * phi)
            p = np.outer(y,y) * phi
        elif self.kernel == 'l':
            p = y[:,None] * x
            p = np.dot(p,np.transpose(p))
            phi = p
        P = matrix(p)
        Q = matrix(-np.ones((size,1)))
        G = matrix(np.vstack((-np.eye(size),np.eye(size))))
        H = matrix(np.vstack((np.zeros((size,1)),np.ones((size,1)))))
        A = matrix(y.reshape(1,size))
        B = matrix(np.zeros(1))
        solvers.options['show_progress'] = False
        sol = solvers.qp(P, Q, G, H, A, B)
        params = np.array(sol['x'])
        params = np.where(params > 1e-4 , params, 0)
        self.params = params
        return phi

    def train(self,features,labels):
        phi = self.get_parameters(features,labels)
        parameters = self.params
        params = parameters
        y = labels
        x = features
        parameters = parameters.reshape(-1)#to change (size,1) to (size,)
        positive_indices = np.where(parameters>0)[0]
        positive_features = x[parameters>0]
        positive_labels = y[parameters>0]
        self.positive_features = positive_features
        self.positive_labels = positive_labels
        params = params[parameters>0]
        if self.kernel == 'g':
            grid = np.ix_(positive_indices,positive_indices)
            sub_phi = phi[grid]
            b = positive_labels - np.sum(params * positive_labels.reshape(params.shape) * sub_phi , axis = 0)
            b = np.mean(b)
            self.b = b
        elif self.kernel == 'l':
            w = np.sum(params * positive_labels.reshape(params.shape) * positive_features , axis = 0)
            wx = w@x.T
            b = positive_labels - np.dot(positive_features,w)
            b = np.mean(b)
            self.w = w
            self.b = b

    def predict(self,test_features,test_labels):
        distances = []
        if self.kernel == 'g':
            parameters = self.params.reshape(-1)
            phi = euclidean_distances(self.positive_features,test_features,squared=True)
            phi = np.exp(-self.gamma * phi)
            prediction = np.sum(self.params[parameters>0] * self.positive_labels.reshape(self.params[parameters>0].shape)*phi, axis = 0) + self.b
        elif self.kernel == 'l':
            prediction = self.w @ test_features.T + self.b
        distances.append(np.absolute(prediction))
        prediction[prediction >= 0] = 1
        prediction[prediction < 0] = -1
        accuracy = accuracy_score(test_labels,prediction)
        return accuracy,prediction,distances


# In[4]:


def write_text_file_sv(filename,value):
    with open(filename,'w') as filehandle:
        for (y,x) in value:
            filehandle.write(str(y)+"\n")
            filehandle.write(str(x))
            filehandle.write("\n")


# In[8]:


def libsvm(kernel,train_features,train_labels,test_features,test_labels):
    prob = svm_problem(train_labels, train_features)
    if kernel == 'l':
        param = svm_parameter('-s 0 -t 0') #-g = gamma = 1/num_of_features = 1/784, default setting
    elif kernel == 'g':
        param = svm_parameter('-s 0 -t 2 -g 0.05') #-g = gamma = 0.05 here
    m = svm_train(prob, param)
    p_labels, p_acc, p_vals = svm_predict(test_labels, test_features, m)


# In[6]:


def train_multiclass(pickle_file,data = data1,gamma=0.05):
    if os.path.exists(pickle_file):
        inp = open(pickle_file,'rb')
        parameters = pickle.load(inp)
        inp.close()
        return parameters
    label = list(range(10))
    pairs = list(combinations(label,2))
    svm_dictionary = {}
    for pair in pairs:
        train_features,train_labels = preprocess_data(data,pair[0],pair[1])
        my_svm = SVM_binaryclass('g',gamma)
        my_svm.train(train_features,train_labels)
        svm_dictionary[pair] = my_svm
        print("Done",pair[0],pair[1])
    output = open(pickle_file, 'wb')
    pickle.dump(pickle_file,output)
    output.close()
    return svm_dictionary

def predict_multiclass(parameters,gamma = 1,data = data2):
    test_labels = data.iloc[:,-1].values
    test_features = data.iloc[:,:-1].values /255
    final_prediction = []
    total_predictions = []
    total_distances = []
    for key in parameters:
        acc,predictions,distances = parameters[key].predict(test_features,test_labels)
        predictions = np.where(predictions==1,key[0],key[1])
        total_predictions.append(predictions)
        total_distances.append(distances[0])
    total_predictions = np.transpose(np.array(total_predictions))
    total_distances = np.transpose(np.array(total_distances))
    for prediction,distance in zip(total_predictions,total_distances):
        pred_data = Counter(prediction)
        list_ = pred_data.most_common()   # Returns all unique items and their counts
        elem1,freq1 = pred_data.most_common(1)[0]
        list_=[elem for (elem,freq) in list_ if freq==freq1]
        list1_ = np.array([np.sum(distance[prediction==elem]) for elem in list_])
        final_prediction.append(list_[np.argmax(list1_)])
    accuracy = accuracy_score(test_labels,final_prediction)
    return accuracy,test_labels,final_prediction
