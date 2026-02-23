"""
Inference Script
Evaluate trained models on test sets
"""

import argparse
import ann.neural_network as nn
from utils.data_loader import load_mnist, load_fashion_mnist
def parse_arguments():
    """
    Parse command-line arguments for inference.
    
    TODO: Implement argparse with: //DONE
    - model_path: Path to saved model weights(do not give absolute path, rather provide relative path)
    - dataset: Dataset to evaluate on
    - batch_size: Batch size for inference
    - hidden_layers: List of hidden layer sizes
    - num_neurons: Number of neurons in hidden layers
    - activation: Activation function ('relu', 'sigmoid', 'tanh')
    """
    parser = argparse.ArgumentParser(description='Run inference on test set')
    parser.add_argument('model_path')
    parser.add_argument('-d','--dataset',required=True,default='mnist',choices=['mnist','fashion_mnist'])
    parser.add_argument('-b','--batch_size',required=True,default=42)
    parser.add_argument('hidden_layers')
    parser.add_argument('num_neurons')
    parser.add_argument('-a','--activation',required=True,default='relu',choices=['relu', 'sigmoid', 'tanh'])
    
    return parser.parse_args()


def load_model(model_path):
    """
    Load trained model from disk.
    """
    pass


def evaluate_model(model, X_test, y_test): 
    """
    Evaluate model on test data.
        
    TODO: Return Dictionary - logits, loss, accuracy, f1, precision, recall
    """
    pass


def main():
    """
    Main inference function.

    TODO: Must return Dictionary - logits, loss, accuracy, f1, precision, recall
    """
    args = parse_arguments()
    
    print("Evaluation complete!")


if __name__ == '__main__':
    main()
