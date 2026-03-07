import numpy as np
from types import SimpleNamespace

from ann.neural_network import NeuralNetwork


def test_backward_propagation(
    eps=1e-5,
    n_w_checks_per_layer=8,
    n_b_checks_per_layer=4,
    seed=42,
):
    """
    Numerical gradient check for backward pass.
    - Verifies gradient shapes match weight/bias shapes
    - Compares analytical grads vs finite-difference grads on sampled params
    """
    rng = np.random.default_rng(seed)
    np.random.seed(seed)

    # Small deterministic config
    args = SimpleNamespace(
        num_layers=2,
        hidden_size=[126, 64],
        activation="relu",
        loss="cross_entropy",
        weight_init="xavier",
        optimizer="sgd",
        learning_rate=1e-3,
        weight_decay=0.0,
        batch_size=32,
        epochs=1,
        dataset="mnist",
    )

    nn = NeuralNetwork(args)

    # Tiny batch (still input_dim=784 as required by your model)
    X = rng.normal(0, 1, size=(4, 784)).astype(np.float64)
    y = np.array([1, 3, 2, 0], dtype=int)
    y_oh = nn._one_hot(y)

    def current_loss():
        logits = nn.forward(X)
        probs = nn.output_activation.activate(logits)
        return nn.loss_fn.loss(y_oh, probs)

    # Analytical gradients
    logits = nn.forward(X)
    probs = nn.output_activation.activate(logits)
    grad_Ws, grad_bs = nn.backward(y_oh, probs)

    # --- shape checks ---
    for i, layer in enumerate(nn.layers):
        assert grad_Ws[i].shape == layer.W.shape, (
            f"Layer {i} grad_W shape mismatch: {grad_Ws[i].shape} vs {layer.W.shape}"
        )
        assert grad_bs[i].shape == layer.b.shape, (
            f"Layer {i} grad_b shape mismatch: {grad_bs[i].shape} vs {layer.b.shape}"
        )

    # --- finite-difference checks ---
    w_errors = []
    b_errors = []

    for li, layer in enumerate(nn.layers):
        # Weight checks
        rows, cols = layer.W.shape
        for _ in range(min(n_w_checks_per_layer, rows * cols)):
            r = rng.integers(0, rows)
            c = rng.integers(0, cols)

            old = layer.W[r, c]

            layer.W[r, c] = old + eps
            l_pos = current_loss()

            layer.W[r, c] = old - eps
            l_neg = current_loss()

            layer.W[r, c] = old  # restore

            num = (l_pos - l_neg) / (2 * eps)
            ana = grad_Ws[li][r, c]
            w_errors.append(abs(num - ana))

        # Bias checks
        bcols = layer.b.shape[1]
        for _ in range(min(n_b_checks_per_layer, bcols)):
            c = rng.integers(0, bcols)

            old = layer.b[0, c]

            layer.b[0, c] = old + eps
            l_pos = current_loss()

            layer.b[0, c] = old - eps
            l_neg = current_loss()

            layer.b[0, c] = old  # restore

            num = (l_pos - l_neg) / (2 * eps)
            ana = grad_bs[li][0, c]
            b_errors.append(abs(num - ana))

    mean_w_err = float(np.mean(w_errors)) if w_errors else 0.0
    mean_b_err = float(np.mean(b_errors)) if b_errors else 0.0
    max_w_err = float(np.max(w_errors)) if w_errors else 0.0
    max_b_err = float(np.max(b_errors)) if b_errors else 0.0

    print("Backward gradient check summary")
    print(f"  mean |dW_num - dW_ana| = {mean_w_err:.6e}")
    print(f"  max  |dW_num - dW_ana| = {max_w_err:.6e}")
    print(f"  mean |db_num - db_ana| = {mean_b_err:.6e}")
    print(f"  max  |db_num - db_ana| = {max_b_err:.6e}")

    # Tighten/relax as needed
    assert mean_w_err < 1e-4, f"Weight gradient mean error too high: {mean_w_err}"
    assert mean_b_err < 1e-4, f"Bias gradient mean error too high: {mean_b_err}"


if __name__ == "__main__":
    test_backward_propagation()
    print("✅ Backward pass gradient check passed.")