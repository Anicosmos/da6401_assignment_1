"""
Optimization Algorithms
Implements: SGD, Momentum, Adam, Nadam, etc.
"""
import numpy as np

class Optimizer:
    """
    Base class for the Optimizer 
    """
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
    
    # def update(self, layer):
    #     """
    #     Update weights and biases based on gradients.
    #     """
    #     layer.weights -= self.learning_rate * layer.grad_w
    #     layer.biases -= self.learning_rate * layer.grad_b

class SGD(Optimizer):
    """
    Stochastic Gradient Descent Optimizer
    """
    def __init__(self, learning_rate=0.01):
        super().__init__(learning_rate)
    
    def update(self, layer):
        """
        Update weights and biases based on gradients.
        """
        layer.weights -= self.learning_rate * layer.grad_w
        layer.biases -= self.learning_rate * layer.grad_b

class Momentum(Optimizer):
    """
    Momentum Optimizer
    """
    def __init__(self, learning_rate=0.01, momentum=0.9):
        super().__init__(learning_rate)
        self.momentum = momentum
        self.velocity_w = None
        self.velocity_b = None
    
    def update(self, layer):
        """
        Update weights and biases based on gradients and momentum.
        """
        if self.velocity_w is None:
            self.velocity_w = np.zeros_like(layer.grad_w)
            self.velocity_b = np.zeros_like(layer.grad_b)
        
        self.velocity_w = self.momentum * self.velocity_w + (1 - self.momentum) * layer.grad_w
        self.velocity_b = self.momentum * self.velocity_b + (1 - self.momentum) * layer.grad_b
        
        layer.weights -= self.learning_rate * self.velocity_w
        layer.biases -= self.learning_rate * self.velocity_b
    
    
class RMSProp(Optimizer):
    """
    RMSProp Optimizer
    """
    def __init__(self, learning_rate=0.01, beta=0.9, epsilon=1e-8):
        super().__init__(learning_rate)
        self.beta = beta
        self.epsilon = epsilon
        self.velocity_w = None
        self.velocity_b = None
    
    def update(self, layer):
        """
        Update weights and biases based on gradients and RMSProp.
        """
        if self.velocity_w is None:
            self.velocity_w = np.zeros_like(layer.grad_w)
            self.velocity_b = np.zeros_like(layer.grad_b)
        
        self.velocity_w = self.beta * self.velocity_w + (1 - self.beta) * (layer.grad_w ** 2)
        self.velocity_b = self.beta * self.velocity_b + (1 - self.beta) * (layer.grad_b ** 2)
        
        layer.weights -= self.learning_rate * layer.grad_w / (np.sqrt(self.velocity_w) + self.epsilon)
        layer.biases -= self.learning_rate * layer.grad_b / (np.sqrt(self.velocity_b) + self.epsilon)
    
class Adam(Optimizer):
    """
    Adam Optimizer
    """
    def __init__(self, learning_rate=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8):
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m_w = None
        self.v_w = None
        self.m_b = None
        self.v_b = None
    
    def update(self, layer):
        """
        Update weights and biases based on gradients and Adam optimization.
        """
        if self.m_w is None:
            self.m_w = np.zeros_like(layer.grad_w)
            self.v_w = np.zeros_like(layer.grad_w)
            self.m_b = np.zeros_like(layer.grad_b)
            self.v_b = np.zeros_like(layer.grad_b)
        
        # Update biased first moment estimate
        self.m_w = self.beta1 * self.m_w + (1 - self.beta1) * layer.grad_w
        self.m_b = self.beta1 * self.m_b + (1 - self.beta1) * layer.grad_b
        
        # Update biased second raw moment estimate
        self.v_w = self.beta2 * self.v_w + (1 - self.beta2) * (layer.grad_w ** 2)
        self.v_b = self.beta2 * self.v_b + (1 - self.beta2) * (layer.grad_b ** 2)
        
        # Compute bias-corrected first moment estimate
        m_hat_w = self.m_w / (1 - self.beta1)
        m_hat_b = self.m_b / (1 - self.beta1)
        
        # Compute bias-corrected second raw moment estimate
        v_hat_w = self.v_w / (1 - self.beta2)
        v_hat_b = self.v_b / (1 - self.beta2)
        
        # Update parameters
        layer.weights -= self.learning_rate * m_hat_w / (np.sqrt(v_hat_w) + self.epsilon)
        layer.biases -= self.learning_rate * m_hat_b / (np.sqrt(v_hat_b) + epsilon)

class Nadam(Optimizer):
    """
    Nadam Optimizer
    """
    def __init__(self, learning_rate=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8):
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m_w = None
        self.v_w = None
        self.m_b = None
        self.v_b = None
    
    def update(self, layer):
        """
        Update weights and biases based on gradients and Nadam optimization.
        """
        if self.m_w is None:
            self.m_w = np.zeros_like(layer.grad_w)
            self.v_w = np.zeros_like(layer.grad_w)
            self.m_b = np.zeros_like(layer.grad_b)
            self.v_b = np.zeros_like(layer.grad_b)
        
        # Update biased first moment estimate
        self.m_w = self.beta1 * self.m_w + (1 - self.beta1) * layer.grad_w
        self.m_b = self.beta1 * self.m_b + (1 - self.beta1) * layer.grad_b
        
        # Update biased second raw moment estimate
        self.v_w = self.beta2 * self.v_w + (1 - self.beta2) * (layer.grad_w ** 2)
        self.v_b = self.beta2 * self.v_b + (1 - self.beta2) * (layer.grad_b ** 2)
        
        # Compute bias-corrected first moment estimate
        m_hat_w = (self.beta1 * self.m_w + (1 - self.beta1) * layer.grad_w) / (1 - self.beta1)
        m_hat_b = (self.beta1 * self.m_b + (1 - self.beta1) * layer.grad_b) / (1 - self.beta1)
        
        # Compute bias-corrected second raw moment estimate
        v_hat_w = self.v_w / (1 - self.beta2)
        v_hat_b = self.v_b / (1 - self.beta2)
        
        # Update parameters
        layer.weights -= self.learning_rate * m_hat_w / (np.sqrt(v_hat_w) + self.epsilon)
        layer.biases -= self.learning_rate * m_hat_b / (np.sqrt(v_hat_b) + self.epsilon)