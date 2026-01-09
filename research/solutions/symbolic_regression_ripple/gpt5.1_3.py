import numpy as np
import warnings

try:
    from pysr import PySRRegressor
    _HAS_PYSR = True
except Exception:
    PySRRegressor = None
    _HAS_PYSR = False

import sympy as sp

ALLOWED_FUNCTIONS = {"sin", "cos", "exp", "log"}
ALLOWED_SYMBOLS = {"x1", "x2"}


def _is_valid_expression(expr_str: str) -> bool:
    try:
        expr = sp.sympify(expr_str)
    except Exception:
        return False
    funcs = expr.atoms(sp.Function)
    for f in funcs:
        name = f.func.__name__
        if name not in ALLOWED_FUNCTIONS:
            return False
    symbols = {str(s) for s in expr.free_symbols}
    if not symbols.issubset(ALLOWED_SYMBOLS):
        return False
    return True


class Solution:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def _fit_baseline_model(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        if X.shape[1] < 2:
            raise ValueError("X must have two columns.")
        x1 = X[:, 0]
        x2 = X[:, 1]

        ones = np.ones_like(x1)

        features = [
            ones,
            x1,
            x2,
            x1 ** 2,
            x2 ** 2,
            x1 * x2,
            np.sin(x1),
            np.cos(x1),
            np.sin(x2),
            np.cos(x2),
            np.sin(2 * x1),
            np.cos(2 * x1),
            np.sin(2 * x2),
            np.cos(2 * x2),
            np.sin(x1 + x2),
            np.cos(x1 + x2),
            np.sin(x1 - x2),
            np.cos(x1 - x2),
        ]

        A = np.column_stack(features)
        coeffs, _, _, _ = np.linalg.lstsq(A, y, rcond=None)

        term_codes = [
            "1",
            "x1",
            "x2",
            "x1**2",
            "x2**2",
            "x1*x2",
            "sin(x1)",
            "cos(x1)",
            "sin(x2)",
            "cos(x2)",
            "sin(2*x1)",
            "cos(2*x1)",
            "sin(2*x2)",
            "cos(2*x2)",
            "sin(x1 + x2)",
            "cos(x1 + x2)",
            "sin(x1 - x2)",
            "cos(x1 - x2)",
        ]

        terms = []
        for coef, code in zip(coeffs, term_codes):
            if not np.isfinite(coef) or abs(coef) < 1e-10:
                continue
            c_str = f"{coef:.12g}"
            if code == "1":
                term_str = c_str
            else:
                term_str = f"({c_str})*({code})"
            terms.append(term_str)

        expression = " + ".join(terms)
        if not expression:
            expression = "0"

        predictions = A @ coeffs
        return expression, predictions

    def solve(self, X: np.ndarray, y: np.ndarray) -> dict:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        n_samples = X.shape[0]

        use_pysr = self.kwargs.get("use_pysr", True) and _HAS_PYSR

        if use_pysr:
            try:
                if n_samples < 2000:
                    niterations = 60
                elif n_samples < 20000:
                    niterations = 40
                else:
                    niterations = 30

                model = PySRRegressor(
                    niterations=niterations,
                    binary_operators=["+", "-", "*", "/", "^"],
                    unary_operators=["sin", "cos", "exp", "log"],
                    populations=15,
                    population_size=33,
                    maxsize=30,
                    procs=0,
                    verbosity=0,
                    progress=False,
                    random_state=42,
                )

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model.fit(X, y, variable_names=["x1", "x2"])

                sym_expr = model.sympy()
                expression = str(sym_expr)

                if not _is_valid_expression(expression):
                    raise ValueError("Disallowed functions in expression")

                predictions = model.predict(X)

                return {
                    "expression": expression,
                    "predictions": predictions.tolist(),
                    "details": {"method": "pysr"},
                }
            except Exception:
                pass

        expression, predictions = self._fit_baseline_model(X, y)
        return {
            "expression": expression,
            "predictions": predictions.tolist(),
            "details": {"method": "baseline"},
        }