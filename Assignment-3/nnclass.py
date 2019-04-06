import numpy as np
from helper import sigmoid,relu,relu_derivative
from sklearn.metrics import accuracy_score, confusion_matrix
import random

class network:
    def __init__(self,num_inputs = 0,layers = 0,batch_size = 0,num_outputs = 0, activation = 'sigmoid'):
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.layers = layers
        self.layers.insert(0,num_inputs)
        self.layers.append(self.num_outputs)
        self.batch_size = batch_size
        self.num_layers = len(self.layers)
        self.biases = [np.random.uniform(-0.5,0.5,y).reshape(y,1)  for y in self.layers[1:]]
        self.weights = [np.random.uniform(-0.5,0.5,x*y).reshape(x,y) for x,y in zip(self.layers[1:],self.layers[:-1])] #weights corresponding to n+1th unit layer in rows and nth unit layer in columns
        self.activation = activation
        self.outputs = []
        self.deltas = []

    def get_output(self,inp): #feed forward
        output = inp
        output = np.array(output)
        self.outputs = []
        self.outputs.append(output)
        #print(output.shape)
        if self.activation == 'sigmoid':
            for i in range(len(self.weights)):
                output = sigmoid(output @ self.weights[i].T + self.biases[i].T)
                self.outputs.append(output)
        elif self.activation == 'relu':
            for i in range(len(self.weights)):
                if i == len(self.weights)-1:
                    output = sigmoid(output @ self.weights[i].T + self.biases[i].T)
                else:
                    output = relu(output @ self.weights[i].T + self.biases[i].T)
                self.outputs.append(output) 
        return output

    def backprop(self,layer_num = None,target = None, layer = 'hidden'):
        if(layer_num < 1):
            return
        if layer == 'output':
            delta = np.multiply((target - self.outputs[layer_num]),(self.outputs[layer_num]*(1-self.outputs[layer_num])))
            self.deltas.insert(0,delta)
        else:
            if self.activation == 'sigmoid':
                delta = np.multiply((self.outputs[layer_num]*(1-self.outputs[layer_num])),self.deltas[0] @ self.weights[layer_num])
                self.deltas.insert(0,delta)
            elif self.activation == 'relu':
                delta = np.multiply(relu_derivative(self.outputs[layer_num]),self.deltas[0] @ self.weights[layer_num])
                self.deltas.insert(0,delta)
        self.backprop(layer_num = layer_num-1)

    def encode_targets(self,targets):
        tar = np.zeros((targets.shape[0],self.num_outputs))
        i = 0
        for target in targets:
            tar[i][int(target)] = 1
            i+=1
        return tar

    def decode_output(self,output):
        res = [np.argmax(row) for row in output]
        return res
    
    def update_weights(self,eta = 1):
        dw = []
        for i in range(len(self.weights)):
            delta_w = (eta/self.batch_size) * (self.deltas[i].T @ self.outputs[i])
            dw.append(delta_w)
        return dw

    def update_biases(self,eta = 1):
        db = []
        for i in range(len(self.biases)):
            delta_b = (eta/self.batch_size) * np.sum(self.deltas[i]).T
            db.append(delta_b)
        return db

    def train(self,train_data,eta = 0.1,epochs = 1):

        for a in range(epochs):
            #print(a+1)
            mini_batches = [train_data[k:k+self.batch_size] for k in range(0, train_data.shape[0], self.batch_size)]
            for mini_batch in mini_batches:
                target = self.encode_targets(mini_batch[:,-1])
                self.get_output(mini_batch[:,:-1])
                self.deltas = []
                self.backprop(layer_num=self.num_layers-1,target=target, layer='output')
                weight_update = self.update_weights(eta)
                bias_update = self.update_biases(eta)
                self.weights = [self.weights[k] + weight_update[k] for k in range(len(self.weights))]
                self.biases = [self.biases[k] + bias_update[k] for k in range(len(self.biases))]
            #print(train_data.shape)
            #print("epoch number", a+1, "completed!!" )
            #metrics = self.predict(train_data)
            #print("Accuracy is",metrics[0])

    def predict(self,test_data):
        y_pred = []
        #print(np.bincount(np.asarray(test_data[:,-1],dtype=int)))
        y_pred = self.get_output(test_data[:,:-1])
        y_pred = self.decode_output(y_pred)
        #print(np.bincount(np.asarray(y_pred,dtype=int)))
        #print(y_pred)
        
        acc = accuracy_score(test_data[:,-1],y_pred)
        conf = confusion_matrix(test_data[:,-1],y_pred)
        return acc*100,conf
    
    def loss(self,train_data):
        y_pred = self.get_output(train_data[:,:-1])
        l = 0.5 * np.mean((y_pred-self.encode_targets(train_data[:,-1]))**2)
        return l