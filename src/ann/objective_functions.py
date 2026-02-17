"""
Loss/Objective Functions and Their Derivatives
Implements: Cross-Entropy, Mean Squared Error (MSE)
"""
import numpy as np
def mse(y_true, y_pred):
    loss = np.mean((y_true-y_pred)**2)
    return loss 
def cross_entropy(y_true,y_pred,multiclass=True):
    if multiclass :
        loss = -1 * np.sum(y_true * np.log(y_pred))
    else : ## Binary Cross Entropy
        loss = (-1/len(y_true)) * np.sum(y_true * np.log(y_pred) + (1-y_true)*np.log(1-y_pred))
    return loss
def mse_derivative(y_true, y_pred):
    grad = 2*(y_pred-y_true)/len(y_true)
    return grad 
def cross_entropy_derivative(y_true, y_pred,multiclass=True):
    if multiclass :
        grad = -1 * (y_true / y_pred)
    else : ## done in class simplified expression for derivative 
        grad =  (y_pred-y_true)/(y_pred*(1-y_pred))
    return grad