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
    CLI arguments for inference:
      model_path          : path to the .npy weights file  (positional)
      -d / --dataset      : mnist | fashion_mnist
      --config_path       : optional path to _config.json (auto-detected if omitted)
      -nhl / --num_layers : number of hidden layers  (required if no config file)
      -sz  / --hidden_size: neurons per hidden layer  (required if no config file)
      -a   / --activation : sigmoid | tanh | relu
      -l   / --loss       : cross_entropy | mean_squared_error
    """
    parser = argparse.ArgumentParser(
        description='Run inference on a trained MLP')

    parser.add_argument('model_path',
                        help='Path to saved model weights (.npy)')
    parser.add_argument('-d', '--dataset',
                        default='mnist', choices=['mnist', 'fashion_mnist'],
                        help='Dataset to evaluate on')
    parser.add_argument('--config_path',
                        default=None,
                        help='Path to model config JSON (auto-detected from model_path if omitted)')
    parser.add_argument('-nhl', '--num_layers',
                        type=int, default=None,
                        help='Number of hidden layers (overrides config file)')
    parser.add_argument('-sz', '--hidden_size',
                        type=int, default=None,
                        help='Neurons per hidden layer (overrides config file)')
    parser.add_argument('-a', '--activation',
                        default=None,
                        choices=['sigmoid', 'tanh', 'relu'],
                        help='Activation function (overrides config file)')
    parser.add_argument('-l', '--loss',
                        default=None,
                        choices=['cross_entropy', 'mean_squared_error', 'mse'],
                        help='Loss function (overrides config file)')

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------

def load_model(model_path, config_override=None):
    """
    Reconstruct a NeuralNetwork from saved weights.

    Strategy:
      1. Look for a companion _config.json next to the .npy file.
      2. Merge / override with any values supplied via config_override dict.
      3. Build a NeuralNetwork with dummy cli_args derived from the config.
      4. Load weights into the network.

    Args:
        model_path      : path to the .npy weights file
        config_override : optional dict of config values to override

    Returns:
        Loaded NeuralNetwork instance
    """
    # ---- locate config ----
    config_path = os.path.splitext(model_path)[0] + '_config.json'
    config = {}
    if os.path.isfile(config_path):
        with open(config_path) as f:
            config = json.load(f)

    if config_override:
        config.update({k: v for k, v in config_override.items() if v is not None})

    # ---- sensible defaults if config is missing/incomplete ----
    config.setdefault('num_layers',    3)
    config.setdefault('hidden_size',   128)
    config.setdefault('activation',    'relu')
    config.setdefault('loss',          'cross_entropy')
    config.setdefault('optimizer',     'adam')
    config.setdefault('learning_rate', 1e-3)
    config.setdefault('weight_decay',  0.0)
    config.setdefault('weight_init',   'xavier')
    config.setdefault('dataset',       'mnist')

    # Normalise loss name
    if config['loss'] == 'mean_squared_error':
        config['loss'] = 'mse'

    # Build a simple namespace so NeuralNetwork(cli_args) works
    cli_args = types.SimpleNamespace(**config)

    # Build network (weights will be overwritten by load())
    nn = NeuralNetwork(cli_args)
    nn.load(model_path)
    return nn


# ---------------------------------------------------------------------------
# Evaluate model
# ---------------------------------------------------------------------------

def evaluate_model(model, X_test, y_test):
    """
    Compute classification metrics for the model on (X_test, y_test).

    Args:
        model  : NeuralNetwork instance with loaded weights
        X_test : (N, 784)  normalised pixel values
        y_test : (N,)      integer class labels

    Returns:
        dict with keys:
            logits      : raw softmax output  (N, 10)
            loss        : scalar cross-entropy / MSE
            accuracy    : overall accuracy
            precision   : macro-average precision
            recall      : macro-average recall
            f1          : macro-average F1-score
            confusion_matrix : (10, 10) numpy array
    """
    y_true = y_test.astype(int)
    y_oh = NeuralNetwork._one_hot(y_true)

    logits = model.forward(X_test)
    y_pred_int = np.argmax(logits, axis=1)

    loss     = float(model.loss_fn.loss(y_oh, logits))
    acc      = accuracy_score(y_true, y_pred_int)
    prec     = precision_score(y_true, y_pred_int, average='macro', zero_division=0)
    rec      = recall_score(y_true, y_pred_int, average='macro', zero_division=0)
    f1       = f1_score(y_true, y_pred_int, average='macro', zero_division=0)
    cm       = confusion_matrix(y_true, y_pred_int)

    return {
        'logits':           logits,
        'loss':             loss,
        'accuracy':         float(acc),
        'precision':        float(prec),
        'recall':           float(rec),
        'f1':               float(f1),
        'confusion_matrix': cm,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_arguments()

    # ---- build config overrides from CLI ----
    override = {}
    if args.num_layers  is not None: override['num_layers']  = args.num_layers
    if args.hidden_size is not None: override['hidden_size'] = args.hidden_size
    if args.activation  is not None: override['activation']  = args.activation
    if args.loss        is not None: override['loss']        = args.loss
    override['dataset'] = args.dataset

    # ---- load model ----
    print(f"Loading model from {args.model_path} ...")
    model = load_model(args.model_path, config_override=override)

    # ---- load test data ----
    print(f"Loading {args.dataset} test split ...")
    if args.dataset == 'mnist':
        _, _, _, _, X_test, y_test = load_mnist()
    else:
        _, _, _, _, X_test, y_test = load_fashion_mnist()

    print(f"  Test samples: {X_test.shape[0]}")

    # ---- evaluate ----
    results = evaluate_model(model, X_test, y_test)

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
