import numpy as np

class Solution:
    def __init__(self, **kwargs):
        pass

    def solve(self, X: np.ndarray, y: np.ndarray) -> dict:
        n_samples = X.shape[0]
        if n_samples == 0:
            expression = "sin(x1 + x2) + (x1 - x2)**2 - 1.5*x1 + 2.5*x2 + 1"
            return {
                "expression": expression,
                "predictions": [],
                "details": {}
            }

        x1 = X[:, 0].astype(float)
        x2 = X[:, 1].astype(float)

        # Basis functions inspired by the McCormick function
        f1 = np.sin(x1 + x2)        # sin(x1 + x2)
        f2 = (x1 - x2) ** 2         # (x1 - x2)^2
        f3 = x1                     # x1
        f4 = x2                     # x2
        f5 = np.ones_like(x1)       # constant

        A = np.column_stack((f1, f2, f3, f4, f5))

        coeffs, _, _, _ = np.linalg.lstsq(A, y, rcond=None)

        # Prune very small coefficients
        max_abs = np.max(np.abs(coeffs)) if np.any(coeffs != 0) else 0.0
        tol = 1e-8 * (1.0 + max_abs)
        clean_coeffs = np.where(np.abs(coeffs) < tol, 0.0, coeffs)

        # Build expression string
        features = [
            ("sin(x1 + x2)", clean_coeffs[0]),
            ("(x1 - x2)**2", clean_coeffs[1]),
            ("x1", clean_coeffs[2]),
            ("x2", clean_coeffs[3]),
            ("CONST", clean_coeffs[4]),
        ]

        terms = []
        active_flags = []
        for form, c in features:
            if c == 0.0:
                active_flags.append(False)
                continue
            active_flags.append(True)
            if form == "CONST":
                term_str = f"{c:.12g}"
            else:
                term_str = f"({c:.12g})*{form}"
            terms.append(term_str)

        if not terms:
            expression = "0"
        else:
            expression = " + ".join(terms)

        # Compute predictions using the cleaned coefficients to match the expression
        y_pred = A @ clean_coeffs

        # Complexity estimation (binary and unary ops) based on active features
        # Feature indices: 0: sin(x1 + x2), 1: (x1 - x2)**2, 2: x1, 3: x2, 4: const
        binary_ops = 0
        unary_ops = 0
        k = sum(active_flags)

        if k > 0:
            if len(active_flags) >= 1 and active_flags[0]:
                # sin(x1 + x2): one '+' inside, one '*' outside
                binary_ops += 2
                unary_ops += 1
            if len(active_flags) >= 2 and active_flags[1]:
                # (x1 - x2)**2 with coefficient: '-', '**', '*'
                binary_ops += 3
            if len(active_flags) >= 3 and active_flags[2]:
                # c*x1
                binary_ops += 1
            if len(active_flags) >= 4 and active_flags[3]:
                # c*x2
                binary_ops += 1
            # constant term contributes no internal ops
            # Top-level additions between k terms
            if k >= 2:
                binary_ops += (k - 1)

        complexity = 2 * binary_ops + unary_ops

        return {
            "expression": expression,
            "predictions": y_pred.tolist(),
            "details": {
                "complexity": int(complexity),
                "coefficients": clean_coeffs.tolist(),
            }
        }