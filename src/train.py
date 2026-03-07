"""
Main Training Script
Entry point for training neural networks with command-line arguments
"""
import argparse
import os
import sys
import wandb
# Ensure the src/ directory is on the path when invoked directly
sys.path.insert(0, os.path.dirname(__file__))
from ann.neural_network import NeuralNetwork
from utils.data_loader import load_mnist, load_fashion_mnist

LOAD = False ## if true , uses the saved model 
import numpy as np
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
                        type=float, default=1e-4,
                        help='Initial learning rate')
    parser.add_argument('-wd','--weight_decay',
                        type=float, default=0.0,
                        help='L2 weight-decay coefficient')
    parser.add_argument('-nhl','--num_layers',
                        type=int, default=3,
                        help='Number of hidden layers')
    parser.add_argument('-sz','--hidden_size',
                        nargs='+', type=int, default=[128,64,32],
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
                        default='da6401_a1_tries',
                        help='project name')
    parser.add_argument('-w_e','--wandb_entity',
                        default=None,
                        help='W&B username)')

    # Where to persist the trained model
    parser.add_argument('-m_s','--model_save_path',
                        # default='./models/working_model1.npy',
                        default='./src/best_model.npy',
                        help='Path to save the trained model (.npy), Change the path to src if its the best model ')

    return parser.parse_args()




def load_model(model_path):
    """
    Load trained model from disk.
    """
    data = np.load(model_path, allow_pickle=True).item()
    return data

def main():
    args = parse_arguments()
    print(f"[DEBUG train.py] args = {vars(args)}")
    # ---- Validate hidden_size ----
    if len(args.hidden_size) == 1:
        # If a single value is provided, repeat it for all layers
        hidden_sizes = args.hidden_size * args.num_layers
    elif len(args.hidden_size) != args.num_layers:
        raise ValueError(f"Number of hidden sizes ({len(args.hidden_size)}) must match the number of layers ({args.num_layers}).")
    else:
        hidden_sizes = args.hidden_size

    print(f"Using hidden sizes: {hidden_sizes}")

    # ---- Load dataset ----
    print(f"Loading {args.dataset} ...")
    if args.dataset == 'mnist':
        X_train, y_train, X_val, y_val, X_test, y_test = load_mnist()
    else:
        X_train, y_train, X_val, y_val, X_test, y_test = load_fashion_mnist()

    print(f"  Train: {X_train.shape}  Val: {X_val.shape}  Test: {X_test.shape}")

    # ---- Initialise W&B ----
    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        config=vars(args),
        # Allow offline / disabled mode when WANDB_MODE=disabled
    )

    # ---- Build & train network ----
    nn = NeuralNetwork(args)

    # --- Load Model if available ---
    if LOAD:
        print(f"Loading model from {args.model_save_path} ...")
        weights = load_model(args.model_save_path)
        nn.set_weights(weights)
    
    # --- Configuration Information ---
    print(f"\nArchitecture:")
    for i, layer in enumerate(nn.layers):
        print(f"  Layer {i}: {layer.input_dim} ---> {layer.n_neurons}  "
              f"({layer.activation_function.activation_type})")

    print(f"\nTraining for {args.epochs} epochs  "
          f"| optimizer={args.optimizer}  lr={args.learning_rate}  "
          f"batch={args.batch_size}\n")



    nn.train(X_train, y_train, X_val, y_val) ## The Main Training Step 

    # ---- Final evaluation on test set ----
    test_metrics = nn.evaluate(X_test, y_test)
    print(f"\nTest  accuracy={test_metrics['accuracy']:.4f}  "
          f"loss={test_metrics['loss']:.4f}\n"
          f"F1-Score ={test_metrics['f1']:.4f}\n"
          f"Recall ={test_metrics['recall']:.4f}\n")

    if wandb.run is not None:
        wandb.log({'test_accuracy': test_metrics['accuracy'],
                   'test_loss':     test_metrics['loss'],
                   'test_f1':       test_metrics['f1'],
                   'test_recall':   test_metrics['recall']})

    # ---- Save model ----
    save_path = args.model_save_path
    weights_saved = nn.get_weights()
    np.save(save_path, weights_saved)

    nn.savejson(save_path.replace('.npy', '.json')) ## Saving the config file as well

    if wandb.run is not None:
        wandb.finish()


if __name__ == '__main__':
    main()
