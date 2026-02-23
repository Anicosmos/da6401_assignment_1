"""
Neural Layer Implementation
Handles weight initialization, forward pass, and gradient computation
"""
## To Accelerate the computations using GPU if available, we can use CuPy which is a library that provides a NumPy-like interface for GPU computations.
## Functionally cupy is same as numpy , but it runs on GPU , and it is needed for limited time constrains 
USE_GPU = False 
if USE_GPU:
    try:
        import cupy as xp
    except ImportError:
        print("CuPy is not installed. Falling back to NumPy.")
        import numpy as xp
else:
    import numpy as xp
    
import numpy as np ## incase the autograder uses numpy instead of xp

class NeuralLayer:
    """
    This will define store all the variables and functions related to a single layer in the ANN neural network implementation
    """
    def __init__(self,layer_index,input_dim,n_neurons,activation_function):
        self.layer_index = layer_index
        self.input_dim = input_dim
        self.n_neurons = n_neurons
        self.activation_function = activation_function
        ## Initializing weights and biases to the correct dimentions and values using the initialize_weights function

        self.weights = None 
        self.biases = None
        ## Store the intermediate values for backpropagation
        self.z = None ## this is the logits before activation 
        self.a = None ## this is the output after actication
        self.grad_w = None ## One of the cached values needed for backprop , needed for assignement 
        self.grad_b = None ## gradient of the bias term, needed for assignement
    def initialize_weights(self,type="random"):
        """
        There are to main methods that is used , random , and Xavier (which is a harmonic mean
        of the number of input and output neurons)
        Reference : - https://d2l.ai/chapter_multilayer-perceptrons/numerical-stability-and-init.html#xavier-initialization
        """
        if type == "random":
            self.weights = xp.random.randn(self.input_dim,self.n_neurons)*0.01 ## multiplying it by 0.01 because it might cause "exploding gradients" if 
            # the initial weights are too large
        elif type == "xavier":
            ## Uniform distribution type of initialization of xavier initialization
            limit = xp.sqrt(6/(self.input_dim+self.n_neurons))
            self.weights = xp.random.uniform(low=-limit, high=limit, size=(self.input_dim,self.n_neurons))
        ### Defining the biases to zeros 
        self.biases = xp.zeros((1,self.n_neurons))

    def activate_forward(self,X):
        """
        Computes the Forard Pass and stores the intermediate values for backprop
        """
        ### Main Function that computes the linear transformation 
        self.z = xp.dot(X,self.weights) + self.biases ## Linear transformation
        ## TO apply the activation function which is what makes DL different from traditional ML
        self.a = self.activation_function.activate(self.z) ## Applying the Non linear activation function
        return self.a
    def activate_derivative(self):
        """
        Computes the derivative of the activation function with respect to the input z, which is needed for backpropagation.
        """
        return self.activation_function.derivative(self.z)
    def backward(self):
        pass
    def add_layers(self,layer):
        """
        This function is used to add a layer to the neural network, and it is used in the neural network class to add layers to the network.
        """
        self.layers.append(layer)

    