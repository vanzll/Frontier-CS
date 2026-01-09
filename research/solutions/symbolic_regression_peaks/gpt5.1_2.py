import numpy as np

try:
    from pysr import PySRRegressor
    _HAVE_PYSR = True
except Exception:
    _HAVE_PYSR = False


class Solution:
    def __init__(self, **kwargs):
        pass

    def _polynomial_baseline(self, X: np.ndarray, y: np.ndarray):
        x1 = X[:, 0]
        x2 = X[:, 1]

        # Polynomial features up to degree 3
        ones = np.ones_like(x1)
        f_x1 = x1
        f_x2 = x2
        f_x1_2 = x1 ** 2
        f_x1_x2 = x1 * x2
        f_x2_2 = x2 ** 2
        f_x1_3 = x1 ** 3
        f_x1_2_x2 = (x1 ** 2) * x2
        f_x1_x2_2 = x1 * (x2 ** 2)
        f_x2_3 = x2 ** 3

        A = np.column_stack([
            ones,
            f_x1,
            f_x2,
            f_x1_2,
            f_x1_x2,
            f_x2_2,
            f_x1_3,
            f_x1_2_x2,
            f_x1_x2_2,
            f_x2_3,
        ])

        coeffs, _, _, _ = np.linalg.lstsq(A, y, rcond=None)

        basis_terms = [
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

        tol = 1e-12
        terms = []
        # Intercept
        if abs(coeffs[0]) > tol:
            terms.append(f"{coeffs[0]:.12g}")

        # Remaining terms
        for coef, basis in zip(coeffs[1:], basis_terms[1:]):
            if abs(coef) <= tol:
                continue
            terms.append(f"{coef:.12g}*{basis}")

        if not terms:
            expression = "0.0"
        else:
            expression = " + ".join(terms)

        predictions = A @ coeffs
        return expression, predictions

    def solve(self, X: np.ndarray, y: np.ndarray) -> dict:
        n_samples, n_features = X.shape
        x1 = X[:, 0]
        x2 = X[:, 1]

        if _HAVE_PYSR:
            try:
                # Choose number of iterations based on dataset size
                if n_samples <= 2000:
                    niterations = 80
                elif n_samples <= 8000:
                    niterations = 60
                else:
                    niterations = 40

                model = PySRRegressor(
                    niterations=niterations,
                    binary_operators=["+", "-", "*", "/", "pow"],
                    unary_operators=["sin", "cos", "exp", "log"],
                    populations=20,
                    population_size=33,
                    maxsize=25,
                    verbosity=0,
                    progress=False,
                    random_state=42,
                )

                model.fit(X, y, variable_names=["x1", "x2"])

                try:
                    best_expr = model.sympy()
                except TypeError:
                    best_expr = model.sympy(0)

                expression = str(best_expr)
                predictions = model.predict(X)

                return {
                    "expression": expression,
                    "predictions": predictions.tolist(),
                    "details": {}
                }
            except Exception:
                # Fall back to polynomial baseline if PySR fails
                expression, predictions = self._polynomial_baseline(X, y)
                return {
                    "expression": expression,
                    "predictions": predictions.tolist(),
                    "details": {}
                }
        else:
            # If PySR is not available, use polynomial baseline
            expression, predictions = self._polynomial_baseline(X, y)
            return {
                "expression": expression,
                "predictions": predictions.tolist(),
                "details": {}
            }