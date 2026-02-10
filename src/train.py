"""
Main Training Script
Entry point for training neural networks with command-line arguments
"""

import argparse

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
    parser = argparse.ArgumentParser(description='Train a neural network')
    parser.add_argument('-d','--dataset',required=True,default='mnist',choices=['mnist','fashion_mnist'])
    parser.add_argument('-e','--epochs',required=True,default=1000)
    parser.add_argument('-b','--batch_size',required=True,default=42)
    parser.add_argument('-lr','--learning_rate',required=True,default=0.001)
    parser.add_argument('-o','--optimizer',required=True,default='sgd',choices=['sgd', 'momentum', 'nag', 'rmsprop', 'adam', 'nadam'])
    parser.add_argument('-sz','--hidden_layers',required=True) #not sure
    parser.add_argument('--num_neurons') # or num_layers ?
    parser.add_argument('-a','--activation',required=True,default='relu',choices=['relu', 'sigmoid', 'tanh'])
    parser.add_argument('-l','--loss',required=True,default='mse',choices=['cross_entropy','mse'])
    parser.add_argument('-w_i','--weight_init',required=True,default='random',choices=['random','xavier']) # not sure need to check documentation/d2l , xavier prevents vanshing gradients 
    parser.add_argument('wand_project')
    parser.add_argument('model_save_path',default='./models')
    return parser.parse_args()


def main():
    """
    Main training function.
    """
    args = parse_arguments()
    # print(args.epochs)
    # print("Training complete!")


if __name__ == '__main__':
    main()
