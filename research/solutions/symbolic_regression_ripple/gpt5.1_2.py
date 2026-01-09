import numpy as np
try:
    from pysr import PySRRegressor
    _PYSR_AVAILABLE = True
except Exception:
    PySRRegressor = None
    _PYSR_AVAILABLE = False


class Solution:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def _fit_pysr(self, X: np.ndarray, y: np.ndarray):
        n_samples = X.shape[0]

        # Subsample for very large datasets to keep runtime reasonable
        max_train_samples = 20000
        if n_samples > max_train_samples:
            rng = np.random.default_rng(42)
            idx = rng.choice(n_samples, size=max_train_samples, replace=False)
            X_train = X[idx]
            y_train = y[idx]
        else:
            X_train = X
            y_train = y

        # Adjust number of iterations based on dataset size
        if n_samples < 2000:
            niterations = 80
        elif n_samples > 40000:
            niterations = 40
        else:
            niterations = 60

        model = PySRRegressor(
            niterations=niterations,
            binary_operators=["+", "-", "*", "/"],
            unary_operators=["sin", "cos"],
            populations=15,
            population_size=33,
            maxsize=25,
            verbosity=0,
            progress=False,
            random_state=42,
        )

        model.fit(X_train, y_train, variable_names=["x1", "x2"])

        best_expr = model.sympy()
        expression = str(best_expr)

        predictions = np.asarray(model.predict(X)).ravel()

        details = {}
        try:
            eqs = getattr(model, "equations_", None)
            if eqs is not None:
                # equations_ is a pandas DataFrame sorted by score
                best_row = eqs.iloc[0]
                if "complexity" in best_row:
                    details["complexity"] = int(best_row["complexity"])
        except Exception:
            pass

        return expression, predictions, details

    def _fit_fallback(self, X: np.ndarray, y: np.ndarray):
        x1 = X[:, 0]
        x2 = X[:, 1]

        r2 = x1 ** 2 + x2 ** 2

        basis_cols = []
        basis_exprs = []

        # Constant term
        basis_cols.append(np.ones_like(x1))
        basis_exprs.append("1")

        # Radial polynomial terms
        basis_cols.append(r2)
        basis_exprs.append("x1**2 + x2**2")

        basis_cols.append(r2 ** 2)
        basis_exprs.append("(x1**2 + x2**2)**2")

        basis_cols.append(r2 ** 3)
        basis_exprs.append("(x1**2 + x2**2)**3")

        # Trigonometric terms on r2
        basis_cols.append(np.sin(r2))
        basis_exprs.append("sin(x1**2 + x2**2)")

        basis_cols.append(np.cos(r2))
        basis_exprs.append("cos(x1**2 + x2**2)")

        basis_cols.append(np.sin(2.0 * r2))
        basis_exprs.append("sin(2.0*(x1**2 + x2**2))")

        basis_cols.append(np.cos(2.0 * r2))
        basis_exprs.append("cos(2.0*(x1**2 + x2**2))")

        # Lower-order Cartesian terms
        basis_cols.append(x1)
        basis_exprs.append("x1")

        basis_cols.append(x2)
        basis_exprs.append("x2")

        basis_cols.append(x1 * x2)
        basis_exprs.append("x1*x2")

        basis_cols.append(x1 ** 2)
        basis_exprs.append("x1**2")

        basis_cols.append(x2 ** 2)
        basis_exprs.append("x2**2")

        # Mixed trig terms
        basis_cols.append(x1 * np.sin(r2))
        basis_exprs.append("x1*sin(x1**2 + x2**2)")

        basis_cols.append(x2 * np.sin(r2))
        basis_exprs.append("x2*sin(x1**2 + x2**2)")

        A = np.column_stack(basis_cols)
        coeffs, _, _, _ = np.linalg.lstsq(A, y, rcond=None)

        terms = []
        predictions = np.zeros_like(y, dtype=float)
        for coef, expr, col in zip(coeffs, basis_exprs, basis_cols):
            if abs(coef) < 1e-8:
                continue
            c_str = f"{coef:.12g}"
            if expr == "1":
                term = f"({c_str})"
            else:
                term = f"({c_str})*({expr})"
            terms.append(term)
            predictions += coef * col

        if not terms:
            expression = "0.0"
            predictions = np.zeros_like(y, dtype=float)
        else:
            expression = " + ".join(terms)

        details = {}
        return expression, predictions, details

    def solve(self, X: np.ndarray, y: np.ndarray) -> dict:
        if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
            X = np.asarray(X)
            y = np.asarray(y)

        if X.ndim != 2 or X.shape[1] != 2:
            raise ValueError("X must be of shape (n_samples, 2)")

        expression = None
        predictions = None
        details = {}

        used_pysr = False
        if _PYSR_AVAILABLE:
            try:
                expression, predictions, details = self._fit_pysr(X, y)
                used_pysr = True
            except Exception:
                used_pysr = False

        if not used_pysr:
            expression, predictions, details = self._fit_fallback(X, y)

        result = {
            "expression": expression,
            "predictions": predictions.tolist() if predictions is not None else None,
            "details": details,
        }
        return result