"""
Activation Functions and Their Derivatives
Implements: ReLU, Sigmoid, Tanh, Softmax
"""
import numpy as np
USE_GPU = False 
if USE_GPU:  # type: ignore
    try:
        import cupy as xp
    except ImportError:
        print("CuPy is not installed. Falling back to NumPy.")
        import numpy as xp
else:
    import numpy as xp
    
import numpy as np ## incase the autograder uses numpy instead of xp
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
        activation = xp.maximum(0,z)
        return activation
    def sigmoid(self,z): ## mainly used in binary classification problems 2B Module Reference 
        activation = 1/(1+xp.exp(-1*z))
        return activation
    def softmax(self,z): ## mainly used in multi-class classification problems 2B Module Reference 
        exp_z = xp.exp(z - xp.max(z)) 
        activation = exp_z / xp.sum(exp_z)
        return activation
    def tanh(self,z):
        activation = xp.tanh(z)
        return activation
    ## Thier derivatives 
    def relu_derivative(self,z): ## This will be a step function that is 1 for z>0 and 0 otherwise
        grad = xp.where(z > 0, 1, 0)
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
    
    
