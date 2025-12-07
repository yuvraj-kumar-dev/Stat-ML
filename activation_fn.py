import numpy as np

def sigmoid(x):
    return (1)/(1+np.exp(-x))

def tanh(x):
    return (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))

def softmax(x):
    exp_n = np.exp(x-np.max(x))
    return exp_n/exp_n.sum(axis=0)

def ReLU(x):
    return np.maximum(0,x)

def Leaky_ReLU(x):
    return np.maximum(0.1*x, x)

print(sigmoid(14))
print(tanh(52))

list1 = [3,2,5]
print(softmax(list1))
print(softmax(list1).sum(axis=0))

print(ReLU(12))
print(Leaky_ReLU(-2))
