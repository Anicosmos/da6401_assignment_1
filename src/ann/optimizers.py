"""Optimization algorithms for training neural networks"""
USE_GPU = False ## True for faster Numpy Training on GPU, False for CPU training (default)
if USE_GPU:
    try:
        import cupy as xp
    except ImportError:
        print("CuPy is not installed. Falling back to NumPy.")
        import numpy as xp
else:
    import numpy as xp
# import numpy as np # backup

class Optimizer:
    def __init__(self, learning_rate=0.01, weight_decay=0.0):
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

    def _apply_weight_decay(self, layer): ## needed for assignment 
        if self.weight_decay > 0:
            layer.grad_w = layer.grad_w + self.weight_decay * layer.weights


class SGD(Optimizer):
    def __init__(self, learning_rate=0.01, weight_decay=0.0):
        super().__init__(learning_rate, weight_decay)

    def update(self, layer):
        self._apply_weight_decay(layer)
        layer.weights -= self.learning_rate * layer.grad_w
        layer.biases -= self.learning_rate * layer.grad_b


class Momentum(Optimizer):
    def __init__(self, learning_rate=0.01, momentum=0.9, weight_decay=0.0):
        super().__init__(learning_rate, weight_decay)
        self.momentum = momentum
        self.velocity_w = None
        self.velocity_b = None

    def update(self, layer):
        self._apply_weight_decay(layer)
        if self.velocity_w is None:
            self.velocity_w = xp.zeros_like(layer.grad_w)
            self.velocity_b = xp.zeros_like(layer.grad_b)

        self.velocity_w = self.momentum * self.velocity_w + layer.grad_w
        self.velocity_b = self.momentum * self.velocity_b + layer.grad_b

        layer.weights -= self.learning_rate * self.velocity_w
        layer.biases -= self.learning_rate * self.velocity_b


class NAG(Optimizer):
    """Nesterov Accelerated Gradient"""
    def __init__(self, learning_rate=0.01, momentum=0.9, weight_decay=0.0):
        super().__init__(learning_rate, weight_decay)
        self.momentum = momentum
        self.velocity_w = None
        self.velocity_b = None

    def update(self, layer):
        self._apply_weight_decay(layer)
        if self.velocity_w is None:
            self.velocity_w = xp.zeros_like(layer.grad_w)
            self.velocity_b = xp.zeros_like(layer.grad_b)

        self.velocity_w = self.momentum * self.velocity_w + layer.grad_w
        self.velocity_b = self.momentum * self.velocity_b + layer.grad_b

        # Nesterov correction
        layer.weights -= self.learning_rate * (self.momentum * self.velocity_w + layer.grad_w)
        layer.biases -= self.learning_rate * (self.momentum * self.velocity_b + layer.grad_b)


class RMSProp(Optimizer):
    def __init__(self, learning_rate=0.01, beta=0.9, epsilon=1e-8, weight_decay=0.0):
        super().__init__(learning_rate, weight_decay)
        self.beta = beta
        self.epsilon = epsilon
        self.velocity_w = None
        self.velocity_b = None

    def update(self, layer):
        self._apply_weight_decay(layer)
        if self.velocity_w is None:
            self.velocity_w = xp.zeros_like(layer.grad_w)
            self.velocity_b = xp.zeros_like(layer.grad_b)

        self.velocity_w = self.beta * self.velocity_w + (1.0 - self.beta) * (layer.grad_w ** 2)
        self.velocity_b = self.beta * self.velocity_b + (1.0 - self.beta) * (layer.grad_b ** 2)

        layer.weights -= self.learning_rate * layer.grad_w / (xp.sqrt(self.velocity_w) + self.epsilon)
        layer.biases -= self.learning_rate * layer.grad_b / (xp.sqrt(self.velocity_b) + self.epsilon)


class Adam(Optimizer):
    """Adam optimizer ( reference d2l.ai)"""
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, weight_decay=0.0):
        super().__init__(learning_rate, weight_decay)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0  # step counter for bias correction
        self.m_w = None
        self.v_w = None
        self.m_b = None
        self.v_b = None

    def update(self, layer):
        self._apply_weight_decay(layer)
        if self.m_w is None:
            self.m_w = xp.zeros_like(layer.grad_w)
            self.v_w = xp.zeros_like(layer.grad_w)
            self.m_b = xp.zeros_like(layer.grad_b)
            self.v_b = xp.zeros_like(layer.grad_b)

        self.t += 1

        self.m_w = self.beta1 * self.m_w + (1.0 - self.beta1) * layer.grad_w
        self.m_b = self.beta1 * self.m_b + (1.0 - self.beta1) * layer.grad_b

        self.v_w = self.beta2 * self.v_w + (1.0 - self.beta2) * (layer.grad_w ** 2)
        self.v_b = self.beta2 * self.v_b + (1.0 - self.beta2) * (layer.grad_b ** 2)

        # Bias correction
        m_hat_w = self.m_w / (1.0 - self.beta1 ** self.t)
        m_hat_b = self.m_b / (1.0 - self.beta1 ** self.t)
        v_hat_w = self.v_w / (1.0 - self.beta2 ** self.t)
        v_hat_b = self.v_b / (1.0 - self.beta2 ** self.t)

        layer.weights -= self.learning_rate * m_hat_w / (xp.sqrt(v_hat_w) + self.epsilon)
        layer.biases -= self.learning_rate * m_hat_b / (xp.sqrt(v_hat_b) + self.epsilon)


class Nadam(Optimizer):
    #TODO: Yet to Implement Nadam optimizer
    pass 