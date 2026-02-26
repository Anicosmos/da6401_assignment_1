"""Neural layer implementation"""
### This representsa he functionality of a single layer , which has n neurons and takes input of dimension input_dim . It also stores the weights and biases of the layer , as well as the gradients during backpropagation . The activate_forward method computes the output of the layer given an input X , while the backward method computes the gradients with respect to the weights and biases given the delta from the next layer . The activate_derivative method computes the derivative of the activation function with respect to z 
# , which is used during backpropagation to compute the delta for this layer .
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


class NeuralLayer: ## This implements the neural layer , stores the gradient and other information per layer 
    def __init__(self, layer_index, input_dim, n_neurons, activation_function):
        self.layer_index = layer_index
        self.input_dim = input_dim
        self.n_neurons = n_neurons
        self.activation_function = activation_function

        self.weights = None
        self.biases = None

        self.input = None
        self.z = None
        self.a = None

        self.grad_W = None
        self.grad_b = None


    def initialize_weights(self, type="random"):
        if type == "random":
            self.weights = xp.random.randn(self.input_dim, self.n_neurons) * 0.01
        elif type == "xavier":
            limit = xp.sqrt(6.0 / (self.input_dim + self.n_neurons))
            self.weights = xp.random.uniform(low=-limit, high=limit,
                                             size=(self.input_dim, self.n_neurons))
        elif type == "zeros":
            self.weights = xp.zeros((self.input_dim, self.n_neurons))
        else:
            raise ValueError(f"Unknown weight init type: {type}")

        self.biases = xp.zeros((1, self.n_neurons))

    def activate_forward(self, X):
        self.input = X
        self.z = xp.dot(X, self.weights) + self.biases
        self.a = self.activation_function.activate(self.z)
        return self.a

    def activate_derivative(self):
        return self.activation_function.derivative(self.z)

    def backward(self, delta):
        self.grad_W = xp.dot(self.input.T, delta)
        self.grad_b = xp.sum(delta, axis=0, keepdims=True) ## This ensures that dimention of grad_b is (1, n_neurons) instead of (n_neurons,) which is important for broadcasting during weight updates
        grad_input = xp.dot(delta, self.weights.T)
        return grad_input
