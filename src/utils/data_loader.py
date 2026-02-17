"""
Data Loading and Preprocessing
Handles MNIST and Fashion-MNIST datasets
"""
from sklearn.datasets import fetch_openml
import numpy as np
from sklearn.model_selection import train_test_split

def load_mnist():
    mnist = fetch_openml('mnist_784', version=1,as_frame=False) ## if as_frame is false it will return an numpy array
    ### Normalizing them 
    # X = mnist.data / 255.0 
    # y = mnist.target.astype(int)
    x = mnist.data
    y = mnist.target#.astype(int)
    return x, y
def load_fashion_mnist():
    fashion_mnist = fetch_openml('Fashion-MNIST', version=1,as_frame=False)
    x = fashion_mnist.data
    y = fashion_mnist.target#.astype(int)
    return x, y
