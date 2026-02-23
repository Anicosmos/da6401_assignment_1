"""
Main Neural Network Model class
Handles forward and backward propagation loops
"""
from .neural_layer import NeuralLayer
from .activations import ActivationFunction
from .objective_functions import ObjectiveFunction
from .optimizers import Optimizer
class NeuralNetwork:
    """
    Main model class that orchestrates the neural network training and inference.
    """
    
    def __init__(self, cli_args):
        """
        Initialize the neural network.

        Args:
            cli_args: Command-line arguments for configuring the network
        """
        self.cli_args = cli_args
        self.layers = []
        ## Set the Activation function and the objective function based on the cli_args

    
    def forward(self, X):
        """
        Forward propagation through all layers.
        
        Args:
            X: Input data
            
        Returns:
            Output logits
        """
        output = X
        for layer in self.layers:
            output = layer.forward(output)
        return output
    
    def backward(self, y_true, y_pred):
        """
        Backward propagation to compute gradients.
        
        Args:
            y_true: True labels
            y_pred: Predicted outputs
            
        Returns:
            return grad_w, grad_b
        """
        grad_output = ObjectiveFunction.derivative(y_true, y_pred)
        ## Now we need to backpropagate this gradient through the layers in reverse order
        for layer in reversed(self.layers):
            grad_output = layer.backward(grad_output, layer.a) ## we need to pass the input to the layer which is the output of the previous layer (which is stored in the layer.a)
        return layer.grad_w, layer.grad_b ## the calculated gradients for the weights and biases of the last layer, which is what we need to update the weights and biases of the last layer using the optimizer, and then we can use these gradients to update the weights and biases of the previous layers as well using the optimizer, which is what makes ANN learning possible through backpropagation.
    
    def update_weights(self):
        """
        Update weights using the optimizer.
        """
        for layer in self.layers:
            # Update weights and biases using the optimizer
            pass
    
    def train(self, X_train, y_train, epochs, batch_size):
        """
        Train the network for specified epochs.
        """
        n_samples = X_train.shape[0]
        for epoch in range(epochs):
            pass
        pass 

    
    def evaluate(self, X, y):
        """
        Evaluate the network on given data.
        Here we use the validation set to evaluate the model during training, and we will use the test set to evaluate the model after training is complete, which is what we will implement in the inference script.
        """
        pass
