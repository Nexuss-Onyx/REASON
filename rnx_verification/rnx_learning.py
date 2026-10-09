"""
RNX Reverse-Mode Proof Harness and Gradient Checking
Implements analytical gradients for the contractive reasoning step,
fine-tuning adapter updates, and verifies against numerical finite differences.
"""

import numpy as np

class RNXLearningHarness:
    def __init__(self, d: int = 16, rank: int = 2, seed: int = 123):
        np.random.seed(seed)
        self.d = d
        self.rank = rank

        # Core parameters
        self.W_rec = np.random.randn(d, d) * 0.1
        self.b_rec = np.random.randn(d) * 0.05
        self.W_in = np.random.randn(d, d) * 0.1
        self.W_out = np.random.randn(d, d) * 0.1
        self.b_out = np.random.randn(d) * 0.05

        # Adapters for reasoning types: U (d, rank), V (d, rank)
        self.U = np.random.randn(d, rank) * 0.05
        self.V = np.random.randn(d, rank) * 0.05

    def forward(self, z_0: np.ndarray, x: np.ndarray, c_val: float = 1.0, alpha: float = 0.4):
        """
        Forward pass for a 1-step trajectory for clean gradient checking:
        W_eff = W_rec + c_val * (U @ V.T)
        h = W_eff @ z_0 + W_in @ x + b_rec
        z_prop = tanh(h)
        z_1 = (1 - alpha) * z_0 + alpha * z_prop
        y = W_out @ z_1 + b_out
        """
        W_eff = self.W_rec + c_val * (self.U @ self.V.T)
        h = W_eff @ z_0 + self.W_in @ x + self.b_rec
        z_prop = np.tanh(h)
        z_1 = (1.0 - alpha) * z_0 + alpha * z_prop
        y = self.W_out @ z_1 + self.b_out
        cache = {
            "z_0": z_0, "x": x, "c_val": c_val, "alpha": alpha,
            "W_eff": W_eff, "h": h, "z_prop": z_prop, "z_1": z_1, "y": y
        }
        return y, cache

    def loss(self, y: np.ndarray, y_target: np.ndarray):
        diff = y - y_target
        return 0.5 * np.sum(diff ** 2), diff

    def backward(self, cache: dict, grad_output: np.ndarray):
        """
        Exact analytical backward pass.
        Returns gradients: dW_rec, db_rec, dW_in, dW_out, db_out, dU, dV
        """
        # y = W_out @ z_1 + b_out
        dW_out = np.outer(grad_output, cache["z_1"])
        db_out = grad_output.copy()
        dz_1 = self.W_out.T @ grad_output

        # z_1 = (1 - alpha) * z_0 + alpha * z_prop
        alpha = cache["alpha"]
        dz_prop = dz_1 * alpha

        # z_prop = tanh(h) => dh = dz_prop * (1 - z_prop^2)
        dh = dz_prop * (1.0 - cache["z_prop"] ** 2)

        # h = W_eff @ z_0 + W_in @ x + b_rec
        db_rec = dh.copy()
        dW_in = np.outer(dh, cache["x"])
        dW_eff = np.outer(dh, cache["z_0"])

        # W_eff = W_rec + c_val * (U @ V.T)
        dW_rec = dW_eff.copy()
        c_val = cache["c_val"]
        # d(U @ V.T) = c_val * dW_eff
        # dU = c_val * dW_eff @ V
        # dV = c_val * dW_eff.T @ U
        dU = c_val * (dW_eff @ self.V)
        dV = c_val * (dW_eff.T @ self.U)

        grads = {
            "dW_rec": dW_rec, "db_rec": db_rec, "dW_in": dW_in,
            "dW_out": dW_out, "db_out": db_out, "dU": dU, "dV": dV
        }
        return grads

    def finite_difference_check(self, eps: float = 1e-6):
        z_0 = np.random.randn(self.d)
        x = np.random.randn(self.d)
        y_target = np.random.randn(self.d)

        y, cache = self.forward(z_0, x)
        l, grad_out = self.loss(y, y_target)
        analytical_grads = self.backward(cache, grad_out)

        rel_errors = {}
        # Check dW_rec
        num_dW_rec = np.zeros_like(self.W_rec)
        for i in range(self.d):
            for j in range(self.d):
                old = self.W_rec[i, j]
                self.W_rec[i, j] = old + eps
                y_p, _ = self.forward(z_0, x)
                l_p, _ = self.loss(y_p, y_target)
                self.W_rec[i, j] = old - eps
                y_m, _ = self.forward(z_0, x)
                l_m, _ = self.loss(y_m, y_target)
                self.W_rec[i, j] = old
                num_dW_rec[i, j] = (l_p - l_m) / (2.0 * eps)
        
        diff_rec = np.linalg.norm(analytical_grads["dW_rec"] - num_dW_rec)
        norm_rec = np.linalg.norm(analytical_grads["dW_rec"]) + np.linalg.norm(num_dW_rec)
        rel_errors["W_rec"] = float(diff_rec / norm_rec)

        # Check dU
        num_dU = np.zeros_like(self.U)
        for i in range(self.d):
            for j in range(self.rank):
                old = self.U[i, j]
                self.U[i, j] = old + eps
                y_p, _ = self.forward(z_0, x)
                l_p, _ = self.loss(y_p, y_target)
                self.U[i, j] = old - eps
                y_m, _ = self.forward(z_0, x)
                l_m, _ = self.loss(y_m, y_target)
                self.U[i, j] = old
                num_dU[i, j] = (l_p - l_m) / (2.0 * eps)
        
        diff_u = np.linalg.norm(analytical_grads["dU"] - num_dU)
        norm_u = np.linalg.norm(analytical_grads["dU"]) + np.linalg.norm(num_dU)
        rel_errors["U"] = float(diff_u / norm_u)

        return rel_errors
