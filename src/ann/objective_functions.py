"""
Loss/Objective Functions and Their Derivatives
Implements: Cross-Entropy, Mean Squared Error (MSE)
"""
import numpy as np
def mse(y_true, y_pred):
    loss = np.mean((y_true-y_pred)**2)
    return loss 
def cross_entropy(y_true,y_pred,multiclass=False):
    if multiclass :
        loss = -1 * np.sum(y_true * np.log(y_pred))
    else : ## Binary Cross Entropy ( M-ulti Label ( Independent Classifications))
        loss = (-1/len(y_true)) * np.sum(y_true * np.log(y_pred) + (1-y_true)*np.log(1-y_pred))
    return loss