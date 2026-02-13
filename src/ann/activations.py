"""
Activation Functions and Their Derivatives
Implements: ReLU, Sigmoid, Tanh, Softmax
"""
import numpy as np

def relu(z): ## most used activation function in deep learning
    activation = np.maximum(0,z)
    return activation
def sigmoid(z): ## mainly used in binary classification problems 2B Module Reference 
    activation = 1/(1+np.exp(-1*z))
    return activation

def softmax(z): ## mainly used in multi-class classification problems 2B Module Reference 
    exp_z = np.exp(z - np.max(z)) 
    activation = exp_z / np.sum(exp_z)
    return activation

#TODO: Tanh 