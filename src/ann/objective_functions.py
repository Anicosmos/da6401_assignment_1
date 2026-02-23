"""
Loss/Objective Functions and Their Derivatives
Implements: Cross-Entropy, Mean Squared Error (MSE)
"""
import numpy as np
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
class ObjectiveFunction:
    def __init__(self, objective_type='mse'):
        self.objective_type = objective_type
    def loss(self,y_true,y_pred):
        if self.objective_type == 'mse':
            return self.mse(y_true,y_pred)
        elif self.objective_type == 'cross_entropy':
            return self.cross_entropy(y_true,y_pred)
        else:
            # print("Unsupported, unknown obj function type")
            raise ValueError("Unsupported")
    def derivative(self,y_true,y_pred):
        if self.objective_type == 'mse':
            return self.mse_derivative(y_true,y_pred)
        elif self.objective_type == 'cross_entropy':
            return self.cross_entropy_derivative(y_true,y_pred)
        else:
            # print("Unsupported, unknown obj function type")
            raise ValueError("Unsupported")
        
    def mse(self,y_true, y_pred):
        loss = np.mean((y_true-y_pred)**2)
        return loss 
    def cross_entropy(self,y_true,y_pred,multiclass=True):
        if multiclass :
            loss = -1 * np.sum(y_true * np.log(y_pred))
        else : ## Binary Cross Entropy
            loss = (-1/len(y_true)) * np.sum(y_true * np.log(y_pred) + (1-y_true)*np.log(1-y_pred))
        return loss
    def mse_derivative(self,y_true, y_pred):
        grad = 2*(y_pred-y_true)/len(y_true)
        return grad 
    def cross_entropy_derivative(self,y_true, y_pred,multiclass=True):
        if multiclass :
            grad = -1 * (y_true / y_pred)
        else : ## done in class simplified expression for derivative 
            grad =  (y_pred-y_true)/(y_pred*(1-y_pred))
        return grad