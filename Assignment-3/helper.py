import numpy as np 

def sigmoid(x):
    sig = 1 / (1 + np.exp(-x))
    return sig

def sigmoid_derivative(x):
    der = sigmoid(x)*(1-sigmoid(x))
    np.clip(der,-1e-7,1e7)
    return der