import numpy as np
from helper import sigmoid,sigmoid_derivative
from sklearn.metrics import accuracy_score

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

    def feed_forward_step(self,inp,layer_num):
        res = self.weights[layer_num] @ inp.reshape(-1,1) + self.biases[layer_num]
        return res

    def get_output(self,inp):
        output = inp
        output = np.array(output)
        self.outputs = []
        self.outputs.append(output)
        for i in range(0,self.num_layers-1):
            output = self.feed_forward_step(output,i)
            self.outputs.append(output.reshape(-1))
        return output

    def backprop(self,layer_num = None,target = None, layer = 'hidden'):
        if(layer_num < 1):
            return
        if(layer == 'output'):
            if self.activation == 'sigmoid':
                delta = (target - self.outputs[layer_num]) * sigmoid_derivative(self.outputs[layer_num])
                #print(delta.shape)
                self.deltas.insert(0,delta)
        else:
            if self.activation == 'sigmoid':
                delta = []
                for i in range(self.layers[layer_num]):
                    h = np.dot(self.weights[layer_num][:,i],self.deltas[0])
                    delta.append(h)
                delta = np.array(delta)
                delta = sigmoid_derivative(self.outputs[layer_num]) * delta
                #print(delta.shape)
                self.deltas.insert(0,delta)
        self.backprop(layer_num = layer_num-1)

    def encode_targets(self,target):
        tar = np.zeros((self.num_outputs,))
        tar[int(target)] = 1
        return tar

    def decode_output(self,output):
        return np.argmax(output)
    
    def update_weights(self,eta = 1):
        dw = []
        for i in range(len(self.weights)):
            columns = []
            for j in range(self.weights[i].shape[1]):
                column = eta * self.deltas[i] * self.outputs[i][j]
                columns.append(column)
            columns = np.array(columns).T
            dw.append(columns)
        '''dw = []
        for i in range(self.num_layers):
            delta_w = eta * self.deltas[i] * self.outputs[i]
            dw.append(delta_w.reshape(-1,1))'''
        return dw

    def update_biases(self,eta = 1):
        db = []
        for i in range(len(self.biases)):
            b = self.deltas[i] * eta
            b = np.array(b)
            db.append(b)
        return db

    def train(self,train_data,eta = 1,epochs = 1):
        for a in range(epochs):
            for i in range(0,train_data.shape[0],self.batch_size):
                weight_update = []
                bias_update = []
                #print(train_data.shape)
                for j in range(i,i+self.batch_size):
                    target = self.encode_targets(train_data[j,-1])
                    self.get_output(train_data[j,:-1])
                    #print("The output for the ",j," example is ", self.outputs)
                    self.deltas = []
                    self.backprop(layer_num=self.num_layers-1,target=target,layer='output')
                    weight_update_new = self.update_weights(eta)
                    bias_update_new = self.update_biases(eta)
                    if len(weight_update) == 0:
                        weight_update = weight_update_new
                    else:
                        weight_update = [weight_update[k] + weight_update_new[k] for k in range(len(self.weights))]
                    if len(bias_update) == 0:
                        bias_update = bias_update_new
                    else:
                        bias_update = [bias_update[k] + bias_update_new[k] for k in range(len(self.biases))]
                self.weights = [self.weights[k] + (weight_update[k]/self.batch_size) for k in range(len(self.weights))]
                self.biases = [self.biases[k] + (bias_update[k]/self.batch_size).reshape(-1,1) for k in range(len(self.biases))]
            print("epoch number", a+1, "completed!!" )
            y_pred,acc = self.predict(train_data)
            print("Accuracy is",acc)

    def predict(self,test_data):
        y_pred = []
        for data in test_data:
            out = self.get_output(data[:-1])
            out = self.decode_output(out)
            y_pred.append(out)
        
        acc = accuracy_score(test_data[:,-1],y_pred)
        return y_pred,acc*100