import numpy as np

try:
    from pysr import PySRRegressor
    _HAS_PYSR = True
except Exception:
    PySRRegressor = None
    _HAS_PYSR = False


class Solution:
    def __init__(self, **kwargs):
        # Allow overriding PySR usage via kwargs; default to True if available.
        self.use_pysr = kwargs.get("use_pysr", True) and _HAS_PYSR

    def _fit_with_pysr(self, X: np.ndarray, y: np.ndarray):
        model = PySRRegressor(
            niterations=70,
            binary_operators=["+", "-", "*", "/"],
            unary_operators=["sin", "cos", "exp", "log"],
            populations=15,
            population_size=50,
            maxsize=25,
            parsimony=1e-4,
            verbosity=0,
            progress=False,
            random_state=42,
        )

        model.fit(X, y, variable_names=["x1", "x2"])

        best_expr = model.sympy()
        expression = str(best_expr)

        try:
            predictions = model.predict(X)
        except Exception:
            predictions = None

        return expression, predictions

    def _fit_with_linear_basis(self, X: np.ndarray, y: np.ndarray):
        x1 = X[:, 0]
        x2 = X[:, 1]

        bases = []
        feats = []

        def add_feat(expr_str, value):
            bases.append(expr_str)
            feats.append(value)

        # Polynomial terms
        add_feat("x1", x1)
        add_feat("x2", x2)
        add_feat("x1**2", x1 ** 2)
        add_feat("x2**2", x2 ** 2)
        add_feat("x1*x2", x1 * x2)
        add_feat("x1**3", x1 ** 3)
        add_feat("x2**3", x2 ** 3)
        add_feat("x1**2*x2", x1 ** 2 * x2)
        add_feat("x1*x2**2", x1 * x2 ** 2)

        r2 = x1 ** 2 + x2 ** 2

        # Trigonometric terms and ripple-like interactions
        add_feat("sin(x1)", np.sin(x1))
        add_feat("sin(x2)", np.sin(x2))
        add_feat("cos(x1)", np.cos(x1))
        add_feat("cos(x2)", np.cos(x2))
        add_feat("sin(x1 + x2)", np.sin(x1 + x2))
        add_feat("cos(x1 + x2)", np.cos(x1 + x2))
        add_feat("sin(x1 - x2)", np.sin(x1 - x2))
        add_feat("cos(x1 - x2)", np.cos(x1 - x2))

        add_feat("sin(x1**2 + x2**2)", np.sin(r2))
        add_feat("cos(x1**2 + x2**2)", np.cos(r2))
        add_feat("x1*sin(x1**2 + x2**2)", x1 * np.sin(r2))
        add_feat("x2*sin(x1**2 + x2**2)", x2 * np.sin(r2))
        add_feat("(x1**2 + x2**2)*sin(x1**2 + x2**2)", r2 * np.sin(r2))
        add_feat("(x1**2 + x2**2)*cos(x1**2 + x2**2)", r2 * np.cos(r2))

        n = X.shape[0]
        k = len(feats)
        A = np.empty((n, k + 1), dtype=float)
        A[:, 0] = 1.0
        for j, f in enumerate(feats):
            A[:, j + 1] = f

        coeffs, _, _, _ = np.linalg.lstsq(A, y, rcond=None)

        c0 = coeffs[0]
        expression_parts = [f"{c0:.12g}"]

        max_coef = np.max(np.abs(coeffs))
        if not np.isfinite(max_coef) or max_coef == 0:
            threshold = 1e-8
        else:
            threshold = 1e-8 * max(1.0, max_coef)

        for coef, expr_str in zip(coeffs[1:], bases):
            if not np.isfinite(coef) or abs(coef) < threshold:
                continue
            if coef >= 0:
                expression_parts.append(f"+ {coef:.12g}*({expr_str})")
            else:
                expression_parts.append(f"- {abs(coef):.12g}*({expr_str})")

        if expression_parts:
            expression = " ".join(expression_parts)
        else:
            expression = "0"

        predictions = A @ coeffs
        return expression, predictions

        # End of _fit_with_linear_basis

    def solve(self, X: np.ndarray, y: np.ndarray) -> dict:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        expression = None
        predictions = None

        if self.use_pysr:
            try:
                expression, predictions = self._fit_with_pysr(X, y)
            except Exception:
                self.use_pysr = False

        if not self.use_pysr:
            expression, predictions = self._fit_with_linear_basis(X, y)

        result = {
            "expression": expression,
            "predictions": None if predictions is None else np.asarray(predictions, dtype=float).tolist(),
            "details": {},
        }
        return result