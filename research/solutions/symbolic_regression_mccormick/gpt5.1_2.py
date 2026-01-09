import numpy as np

class Solution:
    def __init__(self, **kwargs):
        pass

    def solve(self, X: np.ndarray, y: np.ndarray) -> dict:
        x1 = X[:, 0].astype(float)
        x2 = X[:, 1].astype(float)

        # Linear baseline via least squares
        A = np.column_stack([x1, x2, np.ones_like(x1)])
        coeffs, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
        a, b, c = coeffs
        pred_lin = A @ coeffs
        mse_lin = np.mean((y - pred_lin) ** 2)

        # McCormick function candidate
        pred_mcc = np.sin(x1 + x2) + (x1 - x2) ** 2 - 1.5 * x1 + 2.5 * x2 + 1.0
        mse_mcc = np.mean((y - pred_mcc) ** 2)

        if mse_mcc <= mse_lin:
            expression = "sin(x1 + x2) + (x1 - x2)**2 - 1.5*x1 + 2.5*x2 + 1"
            predictions = pred_mcc
        else:
            expression = f"{a:.12g}*x1 + {b:.12g}*x2 + {c:.12g}"
            predictions = pred_lin

        return {
            "expression": expression,
            "predictions": predictions.tolist(),
            "details": {}
        }