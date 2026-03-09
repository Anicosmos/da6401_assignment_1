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
        self.weight_init = getattr(cli_args, 'weight_init', 'xavier')

        #build layers
        self.create_network()

        # ---- instantiate one optimizer per layer ----
        self.optimizers = self.start_optimizers()

    # ------------------------------------------------------------------
    # Network construction
    # ------------------------------------------------------------------
    def _normalize_hidden_sizes(self):
        num_hidden = int(getattr(self.cli_args, "num_layers", 3))
        hs = getattr(self.cli_args, "hidden_size", 128)

        # Handle comma-separated string (from wandb sweeps)
        if isinstance(hs, str):
            hs = [int(x.strip()) for x in hs.split(',')]

        if isinstance(hs, int):
            return [hs] * num_hidden

        if isinstance(hs, (list, tuple)):
            hs = list(hs)
            # Handle nested list [[128,128,128]] — flatten
            if len(hs) > 0 and isinstance(hs[0], (list, tuple)):
                hs = list(hs[0])
            if len(hs) == 1 and num_hidden > 1:
                return hs * num_hidden
            if len(hs) != num_hidden:
                # Truncate or pad to match num_layers
                if len(hs) > num_hidden:
                    return hs[:num_hidden]
                else:
                    return hs + [hs[-1]] * (num_hidden - len(hs))
            return hs

        raise TypeError(f"hidden_size must be int, str, or list, got {type(hs)}") 
    def create_network(self):
        """Create NeuralLayer objects according to the CLI configuration."""
        self.layers = []  # reset layers list
        # num_hidden = int(getattr(self.cli_args, 'num_layers', 3))
        # hidden_size = getattr(self.cli_args, 'hidden_size', 128)
        hidden_size = self._normalize_hidden_sizes()  
        num_hidden = len(hidden_size)
        weight_init = self.weight_init
        
        layer_dims = [INPUT_DIM] + hidden_size + [NUM_CLASSES]
        activations = [self.hidden_activation] * num_hidden + [self.output_activation]

        for idx, (in_dim, out_dim, act) in enumerate(
                zip(layer_dims[:-1], layer_dims[1:], activations)):
            layer = NeuralLayer(idx, in_dim, out_dim, act)
            layer.initialize_weights(weight_init)
            self.layers.append(layer)
        print(f'Length of layers from the network class {len(self.layers)}')

    def start_optimizers(self):
        ### Start an optimizer instance per layer.
        lr = float(getattr(self.cli_args, 'learning_rate', 0.001))
        wd = float(getattr(self.cli_args, 'weight_decay', 0.0))
        opt_name = getattr(self.cli_args, 'optimizer', 'rmsprop').lower()

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
    def forward(self, X,debug=False):
        """
        Forward Pass  X through every layer in order.
        """
        out = X
        if debug:
            print(f"\n=== FORWARD PASS DEBUG ===")
            print(f"Input shape: {out.shape}, min: {out.min():.4f}, max: {out.max():.4f}, mean: {out.mean():.4f}, std: {out.std():.4f}")
            print(f"Has NaN: {np.isnan(out).any()}, Has Inf: {np.isinf(out).any()}")
        for i,layer in enumerate(self.layers):
            if i == len(self.layers)-1 : ## If its  the output layer then dont activate it 
                # print(f"Forward pass through output layer {i} with activation {layer.activation_function.activation_type} (Not Activated yet )")
                out = layer.forward(out, activate=False) 
                if debug:
                    print(f"\nLayer {i} (OUTPUT - {layer.activation_function.activation_type}, NOT activated):")
                    print(f"  Shape: {out.shape}, min: {out.min():.4f}, max: {out.max():.4f}, mean: {out.mean():.4f}, std: {out.std():.4f}")
                    print(f"  Has NaN: {np.isnan(out).any()}, Has Inf: {np.isinf(out).any()}")
                    print(f"  Sample outputs (first 5): {out[0][:min(5, out.shape[1])]}")
            else :
                # print(f"Forward pass through hidden layer {i} with activation {layer.activation_function.activation_type}")
                out = layer.forward(out, activate=True)
                if debug:
                    print(f"\nLayer {i} (HIDDEN - {layer.activation_function.activation_type}):")
                    print(f"  Shape: {out.shape}, min: {out.min():.4f}, max: {out.max():.4f}, mean: {out.mean():.4f}, std: {out.std():.4f}")
                    print(f"  Has NaN: {np.isnan(out).any()}, Has Inf: {np.isinf(out).any()}")
                    print(f"  Dead neurons (zeros): {(out == 0).sum()} / {out.size} ({(out == 0).sum()/out.size*100:.2f}%)")
        if debug:
            print(f"\n=== END FORWARD PASS ===\n")
        return out ## This will only  return the logits only at the final output layer 

    def backward(self, y_true, y_pred_logits, debug=False):
        """
        Backward propagation to compute gradients.

        Args:
            y_true: integer labels or one-hot labels
            y_pred_logits:  raw logits
        """
        batch_size = y_true.shape[0]

        # Convert integer labels to one-hot if needed
        if y_true.ndim == 1 or (y_true.ndim == 2 and y_true.shape[1] == 1):
            y_true_oh = self._one_hot(y_true.flatten().astype(int), num_classes=y_pred_logits.shape[1])
        else:
            y_true_oh = y_true

        # Accept both logits and probabilities for autograder compatibility
        if self._is_probability_distribution(y_pred_logits):
            y_prob = y_pred_logits
            input_kind = "probs"
        else:
            y_prob = self.output_activation.activate(y_pred_logits)
            input_kind = "logits"
            # print('in logits mode')

        if debug:
            print(f"[DEBUG backward] batch_size={batch_size}")
            print(f"[DEBUG backward] y_true shape={y_true_oh.shape}, y_pred shape={y_pred_logits.shape}")
            print(f"[DEBUG backward] input interpreted as {input_kind}")
            print(f"[DEBUG backward] loss_type={self.loss_fn.objective_type}")
            print(f"[DEBUG backward] num_layers={len(self.layers)}")
            for i, layer in enumerate(self.layers):
                print(f"[DEBUG backward] layer {i}: W={layer.W.shape}, b={layer.b.shape}, act={layer.activation_function.activation_type}")

        # Delta for output layer
        if self.loss_fn.objective_type == 'cross_entropy':
            # Softmax + CE combined gradient wrt logits
            delta = (y_prob - y_true_oh) / batch_size
        else:
            # For MSE this matches current project convention (element-wise activation derivative)
            dL_da = self.loss_fn.derivative(y_true_oh, y_prob)
            delta = dL_da * self.layers[-1].activate_derivative()

        if debug:
            print(f"[DEBUG backward] initial delta shape={delta.shape}")
            print(f"[DEBUG backward] initial delta mean={np.mean(np.abs(delta)):.6e}")

        # backprop through output layer
        delta = self.layers[-1].backward(delta)
        if debug:
            print(f"[DEBUG backward] after output layer backward: delta shape={delta.shape}")
            print(f"[DEBUG backward] output layer grad_W shape={self.layers[-1].grad_W.shape}, mean={np.mean(np.abs(self.layers[-1].grad_W)):.6e}")
            print(f"[DEBUG backward] output layer grad_b shape={self.layers[-1].grad_b.shape}, mean={np.mean(np.abs(self.layers[-1].grad_b)):.6e}")

        # propagate through hidden layers
        for i in reversed(range(len(self.layers) - 1)):
            act_deriv = self.layers[i].activate_derivative()
            if debug:
                print(f"[DEBUG backward] layer {i} act_deriv shape={act_deriv.shape}, mean={np.mean(np.abs(act_deriv)):.6e}")
            delta = delta * act_deriv
            if debug:
                print(f"[DEBUG backward] layer {i} delta after act_deriv: mean={np.mean(np.abs(delta)):.6e}")
            
            delta = self.layers[i].backward(delta)
            if debug:
                print(f"[DEBUG backward] layer {i} grad_W shape={self.layers[i].grad_W.shape}, mean={np.mean(np.abs(self.layers[i].grad_W)):.6e}")
                print(f"[DEBUG backward] layer {i} grad_b shape={self.layers[i].grad_b.shape}, mean={np.mean(np.abs(self.layers[i].grad_b)):.6e}")
                print(f"[DEBUG backward] layer {i} delta out shape={delta.shape}, mean={np.mean(np.abs(delta)):.6e}")

        # collect gradients
        self.grad_W = []
        self.grad_b = []
        for layer in self.layers:
            self.grad_W.append(layer.grad_W)
            self.grad_b.append(layer.grad_b)

        # DEBUG: Print final gradient summary
        if debug:
            print(f"[DEBUG backward] === GRADIENT SUMMARY ===")
            for i, (gw, gb) in enumerate(zip(self.grad_W, self.grad_b)):
                print(f"[DEBUG backward] grad_W[{i}] shape={gw.shape}, min={gw.min():.6e}, max={gw.max():.6e}, mean={np.mean(np.abs(gw)):.6e}")
                print(f"[DEBUG backward] grad_b[{i}] shape={gb.shape}, min={gb.min():.6e}, max={gb.max():.6e}, mean={np.mean(np.abs(gb)):.6e}")

        return self.grad_W, self.grad_b


    @staticmethod
    def _is_probability_distribution(x, atol=1e-6):
        """Return True when rows of x look like valid probability distributions."""
        if x.ndim != 2:
            return False
        if np.any(x < -atol) or np.any(x > 1.0 + atol):
            return False
        row_sums = np.sum(x, axis=1)
        return np.allclose(row_sums, 1.0, atol=atol)

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
    def _quick_eval(self, X, y):
        """
        Lightweight evaluation: only loss + accuracy (no sklearn metrics).
        Used during training to avoid timeout.
        """
        y_oh = self._one_hot(y)
        logits = self.forward(X)
        y_pred = self.output_activation.activate(logits)
        loss = self.loss_fn.loss(y_oh, y_pred)
        preds = np.argmax(y_pred, axis=1)
        acc = float(np.mean(preds == y.astype(int)))
        return {'loss': loss, 'accuracy': acc}
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
        # Safe wandb import — autograder may not have it
        wandb = None
        if use_wandb:
            try:
                import wandb as _wandb
                if _wandb.run is not None:
                    wandb = _wandb
            except ImportError:
                pass

        epochs = int(getattr(self.cli_args, 'epochs', 10))
        batch_size = int(getattr(self.cli_args, 'batch_size', 32))
        n_samples = X_train.shape[0]

        # This can handle both integer labels and one-hot
        if y_train.ndim == 1 or (y_train.ndim == 2 and y_train.shape[1] == 1):
            y_train_oh = self._one_hot(y_train.flatten().astype(int))
        else:
            y_train_oh = y_train

        for epoch in range(1, epochs + 1):
            # Shuffle training data each epoch
            perm = np.random.permutation(n_samples)
            X_shuf = X_train[perm]
            y_shuf = y_train_oh[perm]

            epoch_loss = 0.0
            n_batches = 0
            correct = 0

            for start in range(0, n_samples, batch_size):
                X_batch = X_shuf[start:start + batch_size]
                y_batch = y_shuf[start:start + batch_size]

                # Forward
                y_pred_logits = self.forward(X_batch)
                y_pred = self.output_activation.activate(y_pred_logits) ## Huge Mistake in backprop ! 
                
                # Track accuracy from training batches 
                preds = np.argmax(y_pred, axis=1)
                labels = np.argmax(y_batch, axis=1)
                correct += np.sum(preds == labels)

                # Loss for monitoring
                batch_loss = self.loss_fn.loss(y_batch, y_pred)
                epoch_loss += batch_loss
                n_batches += 1

                # Backward + update
                self.backward(y_batch, y_pred_logits) ## This should also return the list of grads
                self.update_weights()

            epoch_loss /= n_batches
            train_acc = correct / n_samples

            # Evaluate on train and val sets
            # train_metrics = self.evaluate(X_train, y_train)
            # val_metrics = self.evaluate(X_val, y_val)
            # Use lightweight eval during training (no sklearn overhead)
            # train_metrics = self._quick_eval(X_train, y_train)

            val_metrics = self._quick_eval(X_val, y_val)

            print(f"Epoch {epoch}/{epochs}  "
                  f"loss={epoch_loss:.4f}  "
                  f"train_acc={train_acc:.4f}  "
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
                    'train_accuracy': train_acc,
                    'val_loss': val_metrics['loss'],
                    'val_accuracy': val_metrics['accuracy'],
                }
                # Log gradient norms for analysis (first + last hidden layer)
                for i, layer in enumerate(self.layers[:-1]):
                    if layer.grad_W is not None:
                        log_dict[f'grad_norm_layer_{i}'] = float(np.linalg.norm(layer.grad_W))
                wandb.log(log_dict)

        return log_dict if epochs > 0 else None
  
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
        y_flat = y.flatten().astype(int)
        print(f'Shape of y in evaluate {y.shape}, flattened to {y_flat.shape}')
        y_oh = self._one_hot(y_flat)
        logits = self.forward(X)
        y_pred = self.output_activation.activate(logits)
        loss = self.loss_fn.loss(y_oh, y_pred)
        preds = np.argmax(y_pred, axis=1)

        ### With the Other Parameters 

        acc      = accuracy_score(y_flat, preds)
        prec     = precision_score(y_flat, preds, average='macro', zero_division=0)
        rec      = recall_score(y_flat, preds, average='macro', zero_division=0)
        f1       = f1_score(y_flat, preds, average='macro', zero_division=0)
        cm       = confusion_matrix(y_flat, preds)

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
        print(len(self.layers))
        for i, layer in enumerate(self.layers):
            d[f"W{i}"] = layer.W.copy()
            d[f"b{i}"] = layer.b.copy()
        return d

    # def set_weights(self, weight_dict):
    #     for i, layer in enumerate(self.layers):
    #         w_key = f"W{i}"
    #         b_key = f"b{i}"
    #         if w_key in weight_dict:
    #             layer.W = weight_dict[w_key].copy()
    #         if b_key in weight_dict:
    #             layer.b = weight_dict[b_key].copy()


    def set_weights(self, weight_dict):
        w_keys = sorted([k for k in weight_dict if k.startswith("W")],
                        key=lambda x: int(x[1:]))
        b_keys = sorted([k for k in weight_dict if k.startswith("b")],
                        key=lambda x: int(x[1:]))
        print(f'Number of layers {len(self.layers)}, Number of keys {(w_keys)}')
        # Check if we need to rebuild the network from weight shapes
        needs_rebuild = (len(self.layers) != len(w_keys))
        if not needs_rebuild:
            for i, layer in enumerate(self.layers):
                if layer.W.shape != weight_dict[f"W{i}"].shape:
                    needs_rebuild = True
                    break

        # if needs_rebuild:
        #     print('Needs Rebuild')
        #     self.layers = []
        #     for i, (wk, bk) in enumerate(zip(w_keys, b_keys)):
        #         W = weight_dict[wk]
        #         b = weight_dict[bk]
        #         in_dim, out_dim = W.shape
        #         # last layer gets output activation, others get hidden activation
        #         act = self.output_activation if i == len(w_keys) - 1 else self.hidden_activation
        #         layer = NeuralLayer(i, in_dim, out_dim, act)
        #         layer.W = W.copy()
        #         layer.b = b.copy()
        #         self.layers.append(layer)
        #     self.optimizers = self.start_optimizers()
        # else:
        #     for i, layer in enumerate(self.layers):
        #         layer.W = weight_dict[f"W{i}"].copy()
        #         layer.b = weight_dict[f"b{i}"].copy()
        for i, layer in enumerate(self.layers):
                layer.W = weight_dict[f"W{i}"].copy()
                layer.b = weight_dict[f"b{i}"].copy()

    # Model serialisation ( Own implementation) 
    def savejson(self, path):
        """
        A companion JSON config is written alongside the .npy file.

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

    