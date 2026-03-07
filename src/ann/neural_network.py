"""
Main Neural Network Model class
Handles layer construction, forward/backward propagation, training and evaluation
"""
import numpy as np
import json
import os

from .neural_layer import NeuralLayer
from .activations import ActivationFunction
from .objective_functions import ObjectiveFunction
from .optimizers import SGD, Momentum, NAG, RMSProp, Adam, Nadam
from sklearn.metrics import (accuracy_score, precision_score,
                              recall_score, f1_score, confusion_matrix)
# We Know the Number of output classes for MNIST as 
NUM_CLASSES = 10 ## As in 10 digits
INPUT_DIM = 784 ## 28x28 flattened input


class NeuralNetwork:
    """
    Configurable Multi-Layer Perceptron for image classification.
    """

    def __init__(self, cli_args):
        """
        Build and initialise the network from parsed CLI arguments.

        Args:
            cli_args: Namespace returned by argparse.parse_args()
        """
        self.cli_args = cli_args
        self.layers = []

        #activation for hidden layers
        activation_type = getattr(cli_args, 'activation', 'relu')
        self.hidden_activation = ActivationFunction(activation_type)

        # Output layer always uses softmax for multi-class classification for our particular problem
        self.output_activation = ActivationFunction('softmax')  

        #loss function
        loss_type = getattr(cli_args, 'loss', 'cross_entropy')
        self.loss_fn = ObjectiveFunction(loss_type)

        #weight initialisation
        self.weight_init = getattr(cli_args, 'weight_init', 'random')

        #build layers
        self.create_network()

        # ---- instantiate one optimizer per layer ----
        self.optimizers = self.start_optimizers()

    # ------------------------------------------------------------------
    # Network construction
    # ------------------------------------------------------------------

    def create_network(self):
        """Create NeuralLayer objects according to the CLI configuration."""
        num_hidden = int(getattr(self.cli_args, 'num_layers', 3))
        hidden_size = getattr(self.cli_args, 'hidden_size', 128)
        weight_init = self.weight_init

        if isinstance(hidden_size, int):
            hidden_size = [hidden_size] * num_hidden
        elif isinstance(hidden_size, list) :
            if len(hidden_size) != num_hidden:
                raise ValueError(f"Number of hidden sizes ({len(hidden_size)}) must match the number of layers ({num_hidden}).")
            hidden_size = hidden_size
        else:
            raise ValueError(f"Invalid hidden_size: {hidden_size}")
        
        layer_dims = [INPUT_DIM] + hidden_size + [NUM_CLASSES]
        activations = [self.hidden_activation] * num_hidden + [self.output_activation]

        for idx, (in_dim, out_dim, act) in enumerate(
                zip(layer_dims[:-1], layer_dims[1:], activations)):
            layer = NeuralLayer(idx, in_dim, out_dim, act)
            layer.initialize_weights(weight_init)
            self.layers.append(layer)

    def start_optimizers(self):
        ### Start an optimizer instance per layer.
        lr = float(getattr(self.cli_args, 'learning_rate', 0.001))
        wd = float(getattr(self.cli_args, 'weight_decay', 0.0))
        opt_name = getattr(self.cli_args, 'optimizer', 'adam').lower()

        optimizer_class = {
            'sgd': SGD,
            'momentum': Momentum,
            'nag': NAG,
            'rmsprop': RMSProp,
            'adam': Adam, ### Used d2l for reference 
            # 'nadam': Nadam, ### Not Implemented yet ## empty Class 
        }.get(opt_name)

        if optimizer_class is None:
            raise ValueError(f"Unknown optimizer: {opt_name}")

        return [optimizer_class(lr, weight_decay=wd) for _ in self.layers]
    
    # Forward pass
    def forward(self, X):
        """
        Forward Pass  X through every layer in order.
        """
        out = X
        for i,layer in enumerate(self.layers):
            if i == len(self.layers)-1 : ## If its  the output layer then dont activate it 
                # print(f"Forward pass through output layer {i} with activation {layer.activation_function.activation_type} (Not Activated yet )")
                out = layer.forward(out, activate=False) 
            else :
                # print(f"Forward pass through hidden layer {i} with activation {layer.activation_function.activation_type}")
                out = layer.forward(out, activate=True)
        return out ## This will only  return the logits only at the final output layer 


    # Backward pass Old Implementation 

    # def backward(self, y_true, _pred):
    #     """
    #     Compute learning gradients via backpropagation.

    #     The output layer uses softmax because its a prediction problem; for cross-entropy loss the combined
    #     gradient simplifies to (_pred - y_true) / N.  For MSE we fall back to the chain rule.

    #     Args:
    #         y_true : one-hot labels  (batch_size, 10)
    #         _pred : network output  (batch_size, 10)

    #     Stores self.grad_W and self.grad_b on every layer.
    #     """
    #     batch_size = y_true.shape[0]

    #     #delta for the output layer
    #     if self.loss_fn.objective_type == 'cross_entropy':
    #         # Softmax + CE combined gradient: dL/dz_out = (ŷ − y) / N
    #         delta = (_pred - y_true) / batch_size
    #     else:
    #         # MSE: chain rule through softmax (element-wise approximation)
    #         dL_da = self.loss_fn.derivative(y_true, _pred)
    #         delta = dL_da * self.layers[-1].activate_derivative()

    #     # backprop through output layer
    #     delta = self.layers[-1].backward(delta)

    #     #propagate through hidden layers
    #     for i in reversed(range(len(self.layers) - 1)):
    #         # multiply by the activation derivative of layer i
    #         delta = delta * self.layers[i].activate_derivative()
    #         delta = self.layers[i].backward(delta)

    #     return self.layers[0].grad_W, self.layers[0].grad_b
    ### Using this Backward prop Function 
    def backward(self, y_true, _pred):
        """
        Backward propagation to compute gradients.
        Returns two numpy arrays: grad_Ws, grad_bs.
        - `grad_Ws[0]` is gradient for the last (output) layer weights,
          `grad_bs[0]` is gradient for the last layer biases, and so on.
        """
        grad_W_list = []
        grad_b_list = []

        # Backprop through layers in reverse; collect grads so that index 0 = last layer
        batch_size = y_true.shape[0]
        #delta for the output layer
        if self.loss_fn.objective_type == 'cross_entropy':
            # Softmax + CE combined gradient: dL/dz_out = (ŷ − y) / N
            delta = (_pred - y_true) / batch_size
        else:
            # MSE: chain rule through softmax (element-wise approximation)
            dL_da = self.loss_fn.derivative(y_true, _pred)
            delta = dL_da * self.layers[-1].activate_derivative()

        # backprop through output layer
        delta = self.layers[-1].backward(delta)
        
        grad_W_list.insert(0, self.layers[-1].grad_W)  # Insert at beginning for correct order
        grad_b_list.insert(0, self.layers[-1].grad_b)
        #propagate through hidden layers
        for i in reversed(range(len(self.layers) - 1)):
            # multiply by the activation derivative of layer i
            delta = delta * self.layers[i].activate_derivative()
            delta = self.layers[i].backward(delta)
            ## Storing the gradients in a list 
            grad_W_list.append(self.layers[i].grad_W)
            grad_b_list.append(self.layers[i].grad_b)

        # create explicit object arrays to avoid numpy trying to broadcast shapes
        self.grad_W = np.empty(len(grad_W_list), dtype=object)
        self.grad_b = np.empty(len(grad_b_list), dtype=object)
        for i, (gw, gb) in enumerate(zip(grad_W_list, grad_b_list)):
            self.grad_W[i] = gw
            self.grad_b[i] = gb

        print("Shape of grad_Ws:", self.grad_W.shape, self.grad_W[1].shape)
        print("Shape of grad_bs:", self.grad_b.shape, self.grad_b[1].shape)
        return self.grad_W, self.grad_b
    # Weight update

    def update_weights(self):
        """Apply the configured optimizer to every layer."""
        for layer, opt in zip(self.layers, self.optimizers):
            opt.update(layer)

    # One-hot encoding helper
    @staticmethod
    def _one_hot(y, num_classes=NUM_CLASSES):
        """Convert integer label array to one-hot matrix."""
        oh = np.zeros((len(y), num_classes))
        oh[np.arange(len(y)), y.astype(int)] = 1.0
        return oh
    ##################################### The Training Loop and Evaluation code ############
    
    def train(self, X_train, y_train, X_val, y_val, use_wandb=True):
        """
        Mini-batch SGD training loop.


        Args:
            X_train : (N, 784)
            y_train : (N,) integer labels
            X_val   : (M, 784)
            y_val   : (M,) integer labels
        """
        import wandb

        epochs = int(getattr(self.cli_args, 'epochs', 10))
        batch_size = int(getattr(self.cli_args, 'batch_size', 32))
        n_samples = X_train.shape[0]

        y_train_oh = self._one_hot(y_train)

        for epoch in range(1, epochs + 1):
            # Shuffle training data each epoch
            perm = np.random.permutation(n_samples)
            X_shuf = X_train[perm]
            y_shuf = y_train_oh[perm]

            epoch_loss = 0.0
            n_batches = 0

            for start in range(0, n_samples, batch_size):
                X_batch = X_shuf[start:start + batch_size]
                y_batch = y_shuf[start:start + batch_size]

                # Forward
                y_pred_logits = self.forward(X_batch)
                _pred = self.output_activation.activate(y_pred_logits)

                # Loss for monitoring
                batch_loss = self.loss_fn.loss(y_batch, _pred)
                epoch_loss += batch_loss
                n_batches += 1

                # Backward + update
                self.backward(y_batch, _pred) ## This should also return the list of grads
                self.update_weights()

            epoch_loss /= n_batches

            # Evaluate on train and val sets
            train_metrics = self.evaluate(X_train, y_train)
            val_metrics = self.evaluate(X_val, y_val)

            print(f"Epoch {epoch}/{epochs}  "
                  f"loss={epoch_loss:.4f}  "
                  f"train_acc={train_metrics['accuracy']:.4f}  "
                  f"val_acc={val_metrics['accuracy']:.4f}  "
                  f"val_loss={val_metrics['loss']:.4f}")
            # log_dict = {
            #         'epoch': epoch,
            #         'train_loss': epoch_loss,
            #         'train_accuracy': train_metrics['accuracy'],
            #         'val_loss': val_metrics['loss'],
            #         'val_accuracy': val_metrics['accuracy'],
            #     }
            # # Log gradient norms for analysis (first + last hidden layer)
            # for i, layer in enumerate(self.layers[:-1]):
            #     if layer.grad_W is not None:
            #         log_dict[f'grad_norm_layer_{i}'] = float(np.linalg.norm(layer.grad_W))
            
            # # W&B logging
            if use_wandb and wandb.run is not None:
                print(f"Logging epoch {epoch} metrics to W&B...")
                log_dict = {
                    'epoch': epoch,
                    'train_loss': epoch_loss,
                    'train_accuracy': train_metrics['accuracy'],
                    'val_loss': val_metrics['loss'],
                    'val_accuracy': val_metrics['accuracy'],
                }
                # Log gradient norms for analysis (first + last hidden layer)
                for i, layer in enumerate(self.layers[:-1]):
                    if layer.grad_W is not None:
                        log_dict[f'grad_norm_layer_{i}'] = float(np.linalg.norm(layer.grad_W))
                wandb.log(log_dict)

        return log_dict ## for wandb logging in the sweep 
  
    # Evaluation

    def evaluate(self, X, y):
        """
        Compute loss and accuracy on a dataset.

        Args:
            X : (N, 784)
            y : (N,) integer class labels

        Returns:
            dict with keys 'loss' and 'accuracy'
        """
        y_oh = self._one_hot(y)
        y_pred = self.forward(X)
        loss = self.loss_fn.loss(y_oh, y_pred)
        preds = np.argmax(y_pred, axis=1)

        ### With the Other Parameters 

        acc      = accuracy_score(y, preds)
        prec     = precision_score(y, preds, average='macro', zero_division=0)
        rec      = recall_score(y, preds, average='macro', zero_division=0)
        f1       = f1_score(y, preds, average='macro', zero_division=0)
        cm       = confusion_matrix(y, preds)

        return {
        'logits':           y_pred,
        'loss':             loss,
        'accuracy':         float(acc),
        'precision':        float(prec),
        'recall':           float(rec),
        'f1':               float(f1),
        'confusion_matrix': cm,
        }
        # return {'loss': float(loss), 'accuracy': float(accuracy)}


    ### Provided by TA 
    def get_weights(self):
        d = {}
        for i, layer in enumerate(self.layers):
            d[f"W{i}"] = layer.W.copy()
            d[f"b{i}"] = layer.b.copy()
        return d

    def set_weights(self, weight_dict):
        for i, layer in enumerate(self.layers):
            w_key = f"W{i}"
            b_key = f"b{i}"
            if w_key in weight_dict:
                layer.W = weight_dict[w_key].copy()
            if b_key in weight_dict:
                layer.b = weight_dict[b_key].copy()
    # Model serialisation ( Own implementation) 
    def savejson(self, path):
        """
        A companion JSON config is written alongside the .npy file.

        Args:
            path : str – destination file path (e.g. '../models/best_model.npy')
        """
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

          # Handle hidden_size dynamically (list or int)
        hidden_size = getattr(self.cli_args, 'hidden_size', 128)
        if isinstance(hidden_size, int):
            hidden_size = [hidden_size] * int(getattr(self.cli_args, 'num_layers', 3))

        # Save config alongside
        config_path = os.path.splitext(path)[0] + '_config.json'
        config = {
            'num_layers':    int(getattr(self.cli_args, 'num_layers', 3)),
            'hidden_size':   hidden_size,
            'activation':    getattr(self.cli_args, 'activation', 'relu'),
            'loss':          getattr(self.cli_args, 'loss', 'cross_entropy'),
            'optimizer':     getattr(self.cli_args, 'optimizer', 'rmsprop'),
            'learning_rate': float(getattr(self.cli_args, 'learning_rate', 0.001)),
            'weight_decay':  float(getattr(self.cli_args, 'weight_decay', 0.0)),
            'weight_init':   getattr(self.cli_args, 'weight_init', 'random'),
            'dataset':       getattr(self.cli_args, 'dataset', 'mnist'),
            'batch_size':    int(getattr(self.cli_args, 'batch_size', 32)),
            'epochs':        int(getattr(self.cli_args, 'epochs', 10)),
        }
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)

        print(f"Model saved to {path}")
        print(f"Config saved to {config_path}")

    # def load(self, path):
    #     """
    #     Load weights from a .npy file previously written by save().

    #     Args:
    #         path : str – path to the .npy file
    #     """
    #     model_data = np.load(path, allow_pickle=True).item()
    #     weights_list = model_data['weights']
    #     biases_list  = model_data['biases']

    #     if len(weights_list) != len(self.layers):
    #         raise ValueError(
    #             f"Mismatch: file has {len(weights_list)} layers "
    #             f"but network has {len(self.layers)} layers.")

    #     for layer, W, b in zip(self.layers, weights_list, biases_list):
    #         layer.weights = W
    #         layer.biases  = b

    #     print(f"Model loaded from {path}")

    