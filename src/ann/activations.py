"""
Activation Functions and Their Derivatives
Implements: ReLU, Sigmoid, Tanh, Softmax
"""
import numpy as np
class ActivationFunction:
    def __init__(self,activation_type='relu'):
        self.activation_type = activation_type
    def activate(self,z):
        if self.activation_type == 'relu':
            return self.relu(z)
        elif self.activation_type == 'sigmoid':
            return self.sigmoid(z)
        elif self.activation_type == 'softmax':
            return self.softmax(z)
        else:
            raise ValueError("Unsupported")
    def derivative(self,z):
        if self.activation_type == 'relu':
            return self.relu_derivative(z)
        elif self.activation_type == 'sigmoid':
            return self.sigmoid_derivative(z)
        elif self.activation_type == 'softmax':
            return self.softmax_derivative(z)
        else:
            raise ValueError("Unsupported")
    def relu(self,z): ## most used activation function in deep learning
        activation = np.maximum(0,z)
        return activation
    def sigmoid(self,z): ## mainly used in binary classification problems 2B Module Reference 
        activation = 1/(1+np.exp(-1*z))
        return activation
    def softmax(self,z): ## mainly used in multi-class classification problems 2B Module Reference 
        exp_z = np.exp(z - np.max(z)) 
        activation = exp_z / np.sum(exp_z)
        return activation
    def tanh(self,z):
        activation = np.tanh(z)
        return activation
    ## Thier derivatives 
    def relu_derivative(self,z): ## This will be a step function that is 1 for z>0 and 0 otherwise
        grad = np.where(z > 0, 1, 0)
        return grad
    def sigmoid_derivative(self,z):
        s = self.sigmoid(z)
        grad = s * (1 - s)
        return grad
    def softmax_derivative(self,z):
        s = self.softmax(z)
        grad = s * (1 - s) 
        return grad
    def tanh_derivative(self,z):
        t = self.tanh(z)
        grad = 1 - t**2
        return grad
    
    
