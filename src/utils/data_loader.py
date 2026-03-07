"""
Data Loading and Preprocessing
Handles MNIST and Fashion-MNIST datasets
"""
# from sklearn.datasets import fetch_openml
from tensorflow.keras.datasets import mnist, fashion_mnist
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np 
def load_mnist():
    # mnist = fetch_openml('mnist_784', version=1,as_frame=False) ## if as_frame is false it will return an numpy array
    # Load MNIST from Keras
    (X_train_val, y_train_val), (X_test, y_test) = mnist.load_data()
    # Flatten images from (28, 28) to (784,)
    X_train_val = X_train_val.reshape(-1, 784).astype(np.float32)
    X_test = X_test.reshape(-1, 784).astype(np.float32)
    ### Normalizing them 
    X = X_train_val / 255.0
    y = y_train_val.astype(int)
    ## Scaling using standar Scaler 
    # scaler = StandardScaler()
    # X = scaler.fit_transform(X)
        # Split train_val into train (90%) and val (10%)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val,
        test_size=0.1,
        random_state=42,
        stratify=y_train_val
    )
    # x = mnist.data
    # y = mnist.target#.astype(int)
    return X_train,y_train,X_val,y_val,X_test,y_test
def load_fashion_mnist():
    # fashion_mnist = fetch_openml('Fashion-MNIST', version=1,as_frame=False)
    # Load Fashion-MNIST from Keras
    (X_train_val, y_train_val), (X_test, y_test) = keras.datasets.fashion_mnist.load_data()
    
    # Flatten images from (28, 28) to (784,)
    X_train_val = X_train_val.reshape(-1, 784).astype(np.float32)
    X_test = X_test.reshape(-1, 784).astype(np.float32)
    # x = fashion_mnist.data
    # y = fashion_mnist.target#.astype(int)
    ### Normalizing them 
    X = X_train_val / 255.0
    y = y_train_val.astype(int)
    # X_train_val, X_test, y_train_val, y_test = train_test_split(
    #     X, y, test_size=10000, random_state=42, stratify=y
    # )
    # X_train,X_val,y_train,y_val = train_test_split(
    #     X_train_val,y_train_val,test_size=0.1,random_state=42,stratify=y_train_val
    # )
        # Split train_val into train (90%) and val (10%)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val,
        test_size=0.1,
        random_state=42,
        stratify=y_train_val
    )
    # x = mnist.data
    # y = mnist.target#.astype(int)
    return X_train,y_train,X_val,y_val,X_test,y_test
