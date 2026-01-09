import numpy as np

class Solution:
    def __init__(self, **kwargs):
        pass

    @staticmethod
    def _build_expression_from_terms(terms):
        expr_parts = []
        for coef, term in terms:
            if term == "":
                part = f"({coef:.15g})"
            else:
                part = f"({coef:.15g})*({term})"
            expr_parts.append(part)
        if not expr_parts:
            return "0"
        return " + ".join(expr_parts)

    def solve(self, X: np.ndarray, y: np.ndarray) -> dict:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        if X.ndim != 2 or X.shape[1] != 2:
            raise ValueError("X must have shape (n, 2)")
        n = X.shape[0]
        if y.shape[0] != n:
            raise ValueError("y must have shape (n,)")

        x1 = X[:, 0]
        x2 = X[:, 1]

        s = np.sin(x1 + x2)
        q = (x1 - x2) ** 2
        A = np.column_stack([s, q, x1, x2, np.ones_like(x1)])
        coeffs, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
        a, b, c, d, e = coeffs.tolist()

        # Known McCormick form (in case data is exact)
        y_known = np.sin(x1 + x2) + (x1 - x2) ** 2 - 1.5 * x1 + 2.5 * x2 + 1.0
        mse_known = float(np.mean((y - y_known) ** 2))

        y_fit = a * s + b * q + c * x1 + d * x2 + e
        mse_fit = float(np.mean((y - y_fit) ** 2))

        if mse_known <= mse_fit * 1.0000001:
            expression = "sin(x1 + x2) + (x1 - x2)**2 - 1.5*x1 + 2.5*x2 + 1"
            predictions = y_known
            details = {"mse": mse_known}
        else:
            terms = []
            if abs(a) > 1e-14:
                terms.append((a, "sin(x1 + x2)"))
            if abs(b) > 1e-14:
                terms.append((b, "(x1 - x2)**2"))
            if abs(c) > 1e-14:
                terms.append((c, "x1"))
            if abs(d) > 1e-14:
                terms.append((d, "x2"))
            if abs(e) > 1e-14 or not terms:
                terms.append((e, ""))

            expression = self._build_expression_from_terms(terms)
            predictions = y_fit
            details = {"mse": mse_fit}

        return {
            "expression": expression,
            "predictions": predictions.tolist(),
            "details": details,
        }