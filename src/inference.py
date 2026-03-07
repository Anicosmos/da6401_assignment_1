"""
Inference Script
Load a trained model and evaluate it on a test set.

Outputs: Accuracy, Precision, Recall, F1-score (per-class and macro-average)
"""
import argparse
import json
import os
import sys
import types

import numpy as np
from sklearn.metrics import (accuracy_score, precision_score,
                              recall_score, f1_score, confusion_matrix)

# Allow direct invocation from the src/ directory
sys.path.insert(0, os.path.dirname(__file__))

from ann.neural_network import NeuralNetwork
from utils.data_loader import load_mnist, load_fashion_mnist


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_arguments():
    """
    Parse command-line arguments.
    
    TODO: Implement argparse with the following arguments: //DONE
    - dataset: 'mnist' or 'fashion_mnist'
    - epochs: Number of training epochs
    - batch_size: Mini-batch size
    - learning_rate: Learning rate for optimizer
    - optimizer: 'sgd', 'momentum', 'nag', 'rmsprop', 'adam', 'nadam'
    - hidden_layers: List of hidden layer sizes
    - num_neurons: Number of neurons in hidden layers
    - activation: Activation function ('relu', 'sigmoid', 'tanh')
    - loss: Loss function ('cross_entropy', 'mse')
    - weight_init: Weight initialization method
    - wandb_project: W&B project name
    - model_save_path: Path to save trained model (do not give absolute path, rather provide relative path)
    """
    parser = argparse.ArgumentParser(description='Train an MLP on MNIST / Fashion-MNIST')

    parser.add_argument('-d','--dataset',
                        default='mnist', choices=['mnist', 'fashion_mnist'],
                        help='This chooses the Dataset to train on , deafult is mnist')
    parser.add_argument('-e','--epochs',
                        type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('-b','--batch_size',
                        type=int, default=32,
                        help='Mini-batch size for Batch SGD')
    parser.add_argument('-l','--loss',
                        default='cross_entropy',
                        choices=['cross_entropy', 'mse'],
                        help='Loss / objective function')
    parser.add_argument('-o','--optimizer',
                        default='rmsprop',
                        choices=['sgd', 'momentum', 'nag', 'rmsprop', 'adam'],
                        help='Optimisation algorithm')
    parser.add_argument('-lr','--learning_rate',
                        type=float, default=1e-3,
                        help='Initial learning rate')
    parser.add_argument('-wd','--weight_decay',
                        type=float, default=0.0,
                        help='L2 weight-decay coefficient')
    parser.add_argument('-nhl','--num_layers',
                        type=int, default=4,
                        help='Number of hidden layers')
    parser.add_argument('-sz','--hidden_size',
                        type=int, default=128,
                        help='Neurons per hidden layer')
    parser.add_argument('-a','--activation',
                        default='relu',
                        choices=['sigmoid', 'tanh', 'relu'],
                        help='Activation function for hidden layers')
    parser.add_argument('-w_i','--weight_init',
                        default='xavier',
                        choices=['random', 'xavier', 'zeros'],
                        help='Weight initialisation strategy')

    # W&B configuration (optional – training works without W&B)
    parser.add_argument('-w_p','--wandb_project',
                        default='da6401_a1',
                        help='project name')
    parser.add_argument('-w_e','--wandb_entity',
                        default=None,
                        help='W&B username)')

    # Where to persist the trained model
    parser.add_argument('-m_s','--model_save_path',
                        default='./models/working_model.npy',
                        help='Path to save the trained model (.npy), Change the path to src if its the best model ')

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------

def load_model(model_path):
    """
    Load trained model from disk.
    """
    data = np.load(model_path, allow_pickle=True).item()
    return data

def load_config(config_path, model_path):
    """
    Load model config JSON.
    If config_path is None, infer from model_path:
      best_model.npy -> best_model_config.json
    """
    if config_path is None:
        config_path = model_path.replace(".npy", "_config.json")

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config JSON not found: {config_path}")

    with open(config_path, "r") as f:
        cfg = json.load(f)
    return cfg



def main():
    args = parse_arguments()
    print(f"[DEBUG train.py] args = {vars(args)}")
    # ---- load only thr test data ----
    print(f"Loading {args.dataset} test split ...")
    if args.dataset == 'mnist':
        _, _, _, _, X_test, y_test = load_mnist()
    else:
        _, _, _, _, X_test, y_test = load_fashion_mnist()

    print(f"  Test samples: {X_test.shape[0]}")

    # ---- Load Config ---
    cfg = load_config(None, args.model_save_path)

    # Force architecture/hparams from saved config (critical)
    for k in ["num_layers", "hidden_size", "activation", "loss",
              "optimizer", "learning_rate", "weight_decay", "weight_init",
              "batch_size", "epochs"]:
        if k in cfg:
            setattr(args, k, cfg[k])

    print(f"Loaded config from JSON: {cfg}")
    # ---- Create Network ---
    nn = NeuralNetwork(args)
    # ---- Load Weights ----
    weights = load_model(args.model_save_path) ## here it is best_model.npy or the model you want to test
    nn.set_weights(weights)

    # --- Configuration Information ---
    print(f"\nArchitecture:")
    for i, layer in enumerate(nn.layers):
        print(f"  Layer {i}: {layer.input_dim} ---> {layer.n_neurons}  "
              f"({layer.activation_function.activation_type})")

    print(f"\nModel was Trained for {args.epochs} epochs  "
          f"| with optimizer={args.optimizer} & lr={args.learning_rate}  "
          f"batch={args.batch_size}\n")

    # ---- evalute ----
    results  = nn.evaluate(X_test, y_test)
    print(f"\n{'='*50}")
    print(f"  Accuracy  : {results['accuracy']:.4f}")
    print(f"  Precision : {results['precision']:.4f}  (macro)")
    print(f"  Recall    : {results['recall']:.4f}  (macro)")
    print(f"  F1-score  : {results['f1']:.4f}  (macro)")
    print(f"  Loss      : {results['loss']:.4f}")
    print(f"{'='*50}\n")

    print("Confusion Matrix:")
    print(results['confusion_matrix'])

    return results


if __name__ == '__main__':
    main()
