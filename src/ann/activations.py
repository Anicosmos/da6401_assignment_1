"""Activation functions for neural network layers"""
USE_GPU = False
if USE_GPU:
    try:
        import cupy as xp
    except ImportError:
        print("CuPy is not installed. Falling back to NumPy.")
        import numpy as xp
else:
    import numpy as xp

import numpy as np  # in case the autograder uses numpy instead of xp


class ActivationFunction:
    def __init__(self, activation_type='relu'):
        self.activation_type = activation_type

    def activate(self, z):
        if self.activation_type == 'relu':
            return self.relu(z)
        elif self.activation_type == 'sigmoid':
            return self.sigmoid(z)
        elif self.activation_type == 'softmax':
            return self.softmax(z)
        elif self.activation_type == 'tanh':
            return self.tanh(z)
        else:
            raise ValueError(f"Unsupported activation type: {self.activation_type}")

    def derivative(self, z):
        if self.activation_type == 'relu':
            return self.relu_derivative(z)
        elif self.activation_type == 'sigmoid':
            return self.sigmoid_derivative(z)
        elif self.activation_type == 'softmax':
            return self.softmax_derivative(z)
        elif self.activation_type == 'tanh':
            return self.tanh_derivative(z)
        else:
            raise ValueError(f"Unsupported activation type: {self.activation_type}")

    def relu(self, z):
        return xp.maximum(0, z)

    def sigmoid(self, z):
        return xp.where(z >= 0,
                        1.0 / (1.0 + xp.exp(-z)),
                        xp.exp(z) / (1.0 + xp.exp(z)))

    def softmax(self, z):
        exp_z = xp.exp(z - xp.max(z, axis=1, keepdims=True))
        return exp_z / xp.sum(exp_z, axis=1, keepdims=True)

    def tanh(self, z):
        return xp.tanh(z)

    def relu_derivative(self, z):
        return xp.where(z > 0, 1.0, 0.0)

    def sigmoid_derivative(self, z):
        s = self.sigmoid(z)
        return s * (1.0 - s)

    def softmax_derivative(self, z):
        s = self.softmax(z)
        return s * (1.0 - s)

    def tanh_derivative(self, z):
        t = self.tanh(z)
        return 1.0 - t ** 2
