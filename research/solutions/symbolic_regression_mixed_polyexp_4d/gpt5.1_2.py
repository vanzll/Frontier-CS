import numpy as np

try:
    from pysr import PySRRegressor
    _HAS_PYSR = True
except Exception:
    _HAS_PYSR = False


class Solution:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def _fit_pysr(self, X: np.ndarray, y: np.ndarray) -> str:
        n_samples = X.shape[0]

        if n_samples > 5000:
            rng = np.random.RandomState(0)
            idx = rng.choice(n_samples, size=5000, replace=False)
            X_train = X[idx]
            y_train = y[idx]
        else:
            X_train, y_train = X, y

        if n_samples <= 2000:
            niterations = self.kwargs.get("niterations", 80)
        elif n_samples <= 10000:
            niterations = self.kwargs.get("niterations", 60)
        else:
            niterations = self.kwargs.get("niterations", 50)

        maxsize = self.kwargs.get("maxsize", 35)

        model = PySRRegressor(
            niterations=niterations,
            binary_operators=["+", "-", "*", "/"],
            unary_operators=["sin", "cos", "exp", "log"],
            maxsize=maxsize,
            populations=15,
            population_size=33,
            progress=False,
            verbosity=0,
            random_state=0,
        )

        model.fit(X_train, y_train, variable_names=["x1", "x2", "x3", "x4"])
        best_expr = model.sympy()
        expression = str(best_expr)
        return expression

    def _fit_baseline(self, X: np.ndarray, y: np.ndarray) -> str:
        x1, x2, x3, x4 = X[:, 0], X[:, 1], X[:, 2], X[:, 3]
        ones = np.ones_like(x1)

        features = [
            ones,
            x1,
            x2,
            x3,
            x4,
            x1 ** 2,
            x2 ** 2,
            x3 ** 2,
            x4 ** 2,
            x1 * x2,
            x1 * x3,
            x1 * x4,
            x2 * x3,
            x2 * x4,
            x3 * x4,
        ]

        e1 = np.exp(-x1 ** 2)
        e2 = np.exp(-x2 ** 2)
        e3 = np.exp(-x3 ** 2)
        e4 = np.exp(-x4 ** 2)

        features.extend([
            e1,
            e2,
            e3,
            e4,
            x1 * e1,
            x2 * e2,
            x3 * e3,
            x4 * e4,
        ])

        A = np.column_stack(features)
        coeffs, _, _, _ = np.linalg.lstsq(A, y, rcond=None)

        feature_exprs = [
            "1",
            "x1",
            "x2",
            "x3",
            "x4",
            "x1**2",
            "x2**2",
            "x3**2",
            "x4**2",
            "x1*x2",
            "x1*x3",
            "x1*x4",
            "x2*x3",
            "x2*x4",
            "x3*x4",
            "exp(-x1**2)",
            "exp(-x2**2)",
            "exp(-x3**2)",
            "exp(-x4**2)",
            "x1*exp(-x1**2)",
            "x2*exp(-x2**2)",
            "x3*exp(-x3**2)",
            "x4*exp(-x4**2)",
        ]

        terms = []
        for c, fexpr in zip(coeffs, feature_exprs):
            if abs(c) < 1e-8:
                continue
            coef_str = f"{c:.12g}"
            if fexpr == "1":
                terms.append(coef_str)
            else:
                terms.append(f"{coef_str}*{fexpr}")

        if not terms:
            expression = "0"
        else:
            expression = " + ".join(terms)

        return expression

    def solve(self, X: np.ndarray, y: np.ndarray) -> dict:
        expression = None

        if _HAS_PYSR:
            try:
                expression = self._fit_pysr(X, y)
            except Exception:
                expression = None

        if expression is None:
            expression = self._fit_baseline(X, y)

        return {
            "expression": expression,
            "predictions": None,
            "details": {}
        }