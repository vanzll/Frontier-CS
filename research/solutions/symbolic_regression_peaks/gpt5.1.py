import numpy as np

class Solution:
    def __init__(self, **kwargs):
        pass

    def solve(self, X: np.ndarray, y: np.ndarray) -> dict:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()

        x1 = X[:, 0]
        x2 = X[:, 1]

        # Basis functions inspired by the classic "peaks" function
        t1 = (1.0 - x1) ** 2 * np.exp(-x1 ** 2 - (x2 + 1.0) ** 2)
        t2 = (x1 / 5.0 - x1 ** 3 - x2 ** 5) * np.exp(-x1 ** 2 - x2 ** 2)
        t3 = np.exp(-(x1 + 1.0) ** 2 - x2 ** 2)
        ones = np.ones_like(x1)

        A = np.column_stack([t1, t2, t3, ones])

        coeffs, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
        c1, c2, c3, c0 = coeffs

        preds = A @ coeffs

        def fmt(c):
            c = float(c)
            if abs(c) < 1e-12:
                c = 0.0
            return repr(c)

        expression = (
            f"{fmt(c1)}*(1 - x1)**2*exp(-x1**2 - (x2 + 1)**2)"
            f" + {fmt(c2)}*(x1/5 - x1**3 - x2**5)*exp(-x1**2 - x2**2)"
            f" + {fmt(c3)}*exp(-(x1 + 1)**2 - x2**2)"
            f" + {fmt(c0)}"
        )

        return {
            "expression": expression,
            "predictions": preds.tolist(),
            "details": {}
        }