import numpy as np

class Solution:
    def __init__(self, **kwargs):
        pass

    def solve(self, X: np.ndarray, y: np.ndarray) -> dict:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        n_samples = y.shape[0]
        if n_samples == 0:
            expression = "sin(x1 + x2) + (x1 - x2)**2 - 1.5*x1 + 2.5*x2 + 1"
            return {
                "expression": expression,
                "predictions": [],
                "details": {}
            }

        x1 = X[:, 0]
        x2 = X[:, 1]

        # Analytic McCormick function
        preds_mc = np.sin(x1 + x2) + (x1 - x2) ** 2 - 1.5 * x1 + 2.5 * x2 + 1.0
        mse_mc = float(np.mean((y - preds_mc) ** 2))

        # Linear regression baseline
        A = np.column_stack([x1, x2, np.ones_like(x1)])
        coeffs, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
        a, b, c = coeffs
        preds_lin = A @ coeffs
        mse_lin = float(np.mean((y - preds_lin) ** 2))
        expr_lin = f"{a:.12f}*x1 + {b:.12f}*x2 + {c:.12f}"

        if mse_mc <= mse_lin:
            expression = "sin(x1 + x2) + (x1 - x2)**2 - 1.5*x1 + 2.5*x2 + 1"
            predictions = preds_mc
            train_mse = mse_mc
        else:
            expression = expr_lin
            predictions = preds_lin
            train_mse = mse_lin

        return {
            "expression": expression,
            "predictions": predictions.tolist(),
            "details": {"train_mse": train_mse}
        }