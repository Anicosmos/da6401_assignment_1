import numpy as np
import argparse
from sklearn.metrics import f1_score
from ann.neural_network import NeuralNetwork
from utils.data_loader import load_mnist, load_fashion_mnist
import json
import os

# best_config= argparse.Namespace(
#             dataset="mnist",
#             epochs=2,
#             batch_size=64,
#             loss="cross_entropy",
#             optimizer="sgd",
#             weight_decay=0.0,
#             learning_rate=0.01,
#             num_layers=2,
#             hidden_size=[64, 64],
#             activation="relu",
#             weight_init="xavier"
#         )
    # ---- Load Config ---
def load_config(config_path, model_path):
    """
    Load model config JSON.
    If config_path is None, infer from model_path:
      best_model.npy -> best_model_config.json
    """
    if config_path is None:
        config_path = model_path.replace("_model.npy", "_config.json")

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config JSON not found: {config_path}")

    with open(config_path, "r") as f:
        cfg = json.load(f)
    return cfg
# cfg = load_config(None, "src/best_model_config.json")

#     # Force architecture/hparams from saved config (critical)
# for k in ["num_layers", "hidden_size", "activation", "loss",
#               "optimizer", "learning_rate", "weight_decay", "weight_init",
#               "batch_size", "epochs"]:
#         if k in cfg:
#             setattr(best_config, k, cfg[k])
best_config = {}#argparse.Namespace(**cfg)
model = NeuralNetwork(best_config)
print(f'Layers is {len(model.layers)}')
weights = np.load("src/best_model.npy", allow_pickle=True).item()
# print(weights)

model.set_weights(weights)
print(f'Layers is {len(model.layers)}')
X_test = np.random.rand(100, 784)  # 100 samples, 784 features

y_true = np.random.randint(0, 10, size=(100,))  # 100 samples, 10 classes (0-9)

dataset = 'mnist'  # or 'fashion_mnist'
if dataset == 'mnist':
        _, _, _, _, X_test, y_true = load_mnist()
else:
        _, _, _, _, X_test, y_true = load_fashion_mnist()

y_pred= model.forward(X_test)

y_pred_labels = np.argmax(y_pred, axis=1)

print("F1 Score:", f1_score(y_true, y_pred_labels, average='macro'))

weights_saved = model.get_weights()
# print(weights_saved)