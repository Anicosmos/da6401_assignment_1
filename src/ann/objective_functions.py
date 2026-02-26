"""Loss functions"""
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

class ObjectiveFunction:
    def __init__(self, objective_type='cross_entropy'):
        self.objective_type = objective_type
    def loss(self, y_true, y_pred):
        if self.objective_type == 'mse':
            return self.mse(y_true, y_pred)
        elif self.objective_type == 'cross_entropy':
            return self.cross_entropy(y_true, y_pred)
        else:
            raise ValueError(f"Unsupported objective type: {self.objective_type}")
    def derivative(self, y_true, y_pred):
        if self.objective_type == 'mse':
            return self.mse_derivative(y_true, y_pred)
        elif self.objective_type == 'cross_entropy':
            return self.cross_entropy_derivative(y_true, y_pred)
        else:
            raise ValueError(f"Unsupported objective type: {self.objective_type}")
    def mse(self, y_true, y_pred):
        return xp.mean((y_true - y_pred) ** 2)
    def cross_entropy(self, y_true, y_pred):
        y_pred_clipped = xp.clip(y_pred, 1e-10, 1.0)
        return -xp.mean(xp.sum(y_true * xp.log(y_pred_clipped), axis=1))
    def mse_derivative(self, y_true, y_pred):
        batch_size = y_true.shape[0]
        return 2.0 * (y_pred - y_true) / batch_size
    def cross_entropy_derivative(self, y_true, y_pred):
        y_pred_clipped = xp.clip(y_pred, 1e-10, 1.0)
        batch_size = y_true.shape[0]
        return -(y_true / y_pred_clipped) / batch_size