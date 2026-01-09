import numpy as np

try:
    from pysr import PySRRegressor
    _HAVE_PYSR = True
except Exception:
    PySRRegressor = None
    _HAVE_PYSR = False


def _estimate_complexity(expression: str) -> int:
    expr = expression.replace(" ", "")
    binary_ops = 0
    i = 0
    while i < len(expr):
        if expr.startswith("**", i):
            binary_ops += 1
            i += 2
        elif expr[i] in "+-*/":
            binary_ops += 1
            i += 1
        else:
            i += 1
    unary_ops = (
        expression.count("sin(")
        + expression.count("cos(")
        + expression.count("exp(")
        + expression.count("log(")
    )
    return 2 * binary_ops + unary_ops


class Solution:
    def __init__(self, **kwargs):
        self.niterations = kwargs.get("niterations", 40)
        self.populations = kwargs.get("populations", 15)
        self.population_size = kwargs.get("population_size", 33)
        self.maxsize = kwargs.get("maxsize", 25)
        self.random_state = kwargs.get("random_state", 42)

    def _polynomial_baseline(self, X: np.ndarray, y: np.ndarray):
        x1 = X[:, 0]
        x2 = X[:, 1]

        ones = np.ones_like(x1)
        feats = [
            ones,            # 1
            x1,              # x1
            x2,              # x2
            x1 * x1,         # x1**2
            x1 * x2,         # x1*x2
            x2 * x2,         # x2**2
            x1 ** 3,         # x1**3
            (x1 ** 2) * x2,  # x1**2*x2
            x1 * (x2 ** 2),  # x1*x2**2
            x2 ** 3,         # x2**3
        ]
        terms = [
            "1",
            "x1",
            "x2",
            "x1**2",
            "x1*x2",
            "x2**2",
            "x1**3",
            "x1**2*x2",
            "x1*x2**2",
            "x2**3",
        ]

        A = np.column_stack(feats)
        coeffs, _, _, _ = np.linalg.lstsq(A, y, rcond=None)

        expr_terms = []
        for coef, term in zip(coeffs, terms):
            if abs(coef) < 1e-12:
                continue
            if term == "1":
                term_str = f"{coef:.12g}"
            else:
                term_str = f"{coef:.12g}*{term}"
            expr_terms.append(term_str)

        if not expr_terms:
            expression = "0.0"
        else:
            expression = " + ".join(expr_terms)

        y_pred = A @ coeffs
        return expression, y_pred

    def solve(self, X: np.ndarray, y: np.ndarray) -> dict:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()

        n = X.shape[0]
        expression = None
        predictions = None
        used_pysr = False

        if _HAVE_PYSR:
            try:
                max_samples = 6000
                if n > max_samples:
                    rng = np.random.RandomState(self.random_state)
                    idx = rng.choice(n, size=max_samples, replace=False)
                    X_train = X[idx]
                    y_train = y[idx]
                else:
                    X_train = X
                    y_train = y

                model = PySRRegressor(
                    niterations=self.niterations,
                    binary_operators=["+", "-", "*", "/"],
                    unary_operators=["sin", "cos", "exp", "log"],
                    populations=self.populations,
                    population_size=self.population_size,
                    maxsize=self.maxsize,
                    verbosity=0,
                    progress=False,
                    random_state=self.random_state,
                )
                model.fit(X_train, y_train, variable_names=["x1", "x2"])

                best_expr_sympy = model.sympy()
                if best_expr_sympy is None:
                    raise RuntimeError("PySR did not return an expression.")

                expression = str(best_expr_sympy)
                preds = model.predict(X)
                predictions = np.asarray(preds, dtype=float).ravel().tolist()
                used_pysr = True
            except Exception:
                used_pysr = False

        if not used_pysr:
            expression, y_pred = self._polynomial_baseline(X, y)
            predictions = np.asarray(y_pred, dtype=float).ravel().tolist()

        details = {}
        if expression is not None:
            details["complexity"] = _estimate_complexity(expression)

        return {
            "expression": expression,
            "predictions": predictions,
            "details": details,
        }