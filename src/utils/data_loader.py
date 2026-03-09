"""
Data Loading and Preprocessing
Handles MNIST and Fashion-MNIST datasets
"""
from tensorflow.keras.datasets import mnist, fashion_mnist
from sklearn.model_selection import train_test_split
import numpy as np


def _split_and_normalize(X_train_val, y_train_val, X_test, y_test, val_size=0.1, random_state=42):
    """Split train/val and normalize all feature splits to [0, 1]."""
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=val_size,
        random_state=random_state,
        stratify=y_train_val
    )

    X_train = X_train.astype(np.float32) / 255.0
    X_val = X_val.astype(np.float32) / 255.0
    X_test = X_test.astype(np.float32) / 255.0

    y_train = y_train.astype(int)
    y_val = y_val.astype(int)
    y_test = y_test.astype(int)

    return X_train, y_train, X_val, y_val, X_test, y_test


def load_mnist():
    # Load MNIST from Keras
    (X_train_val, y_train_val), (X_test, y_test) = mnist.load_data()

    # Flatten images from (28, 28) to (784,)
    X_train_val = X_train_val.reshape(-1, 784)
    X_test = X_test.reshape(-1, 784)

    return _split_and_normalize(X_train_val, y_train_val, X_test, y_test)


def load_fashion_mnist():
    # Load Fashion-MNIST from Keras
    (X_train_val, y_train_val), (X_test, y_test) = fashion_mnist.load_data()

    # Flatten images from (28, 28) to (784,)
    X_train_val = X_train_val.reshape(-1, 784)
    X_test = X_test.reshape(-1, 784)

    return _split_and_normalize(X_train_val, y_train_val, X_test, y_test)
