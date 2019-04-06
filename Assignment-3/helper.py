import numpy as np

def sigmoid(x):
    sig = np.where(x<0,1-1 / (1 + np.exp(x)),1/(1+np.exp(-x)))
    return sig

def relu(x):
    rel = np.where(x<0,0,x)
    return rel

def relu_derivative(x):
    rel_der = np.where(x<=0,0,1)
    return rel_der