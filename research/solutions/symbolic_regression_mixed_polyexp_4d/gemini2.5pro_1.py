import numpy as np
from pysr import PySRRegressor
import sympy

class Solution:
    def __init__(self, **kwargs):
        pass

    def solve(self, X: np.ndarray, y: np.ndarray) -> dict:
        """
        Finds a symbolic expression for the given data using PySR.
        """
        # PySR regressor with parameters tuned for a 4D problem on an 8-core CPU.
        model = PySRRegressor(
            niterations=120,
            populations=24,
            population_size=50,
            binary_operators=["+", "-", "*", "/"],
            unary_operators=["sin", "cos", "exp", "log"],
            maxsize=35,
            procs=8,
            random_state=42,
            verbosity=0,
            progress=False,
            # Constraints can help guide the search to more plausible expressions
            nested_constraints={"exp": {"exp": 0, "log": 0}, "log": {"exp": 0, "log": 0},
                                "cos": {"cos": 0}, "sin": {"sin": 0}},
            # Stop early if a very good expression is found
            early_stop_condition="f(loss, complexity) < 1e-7",
            # A timeout is a crucial safeguard in a timed environment
            timeout_in_seconds=720  # 12 minutes
        )

        try:
            model.fit(X, y, variable_names=["x1", "x2", "x3", "x4"])

            if not hasattr(model, 'equations_') or model.equations_.empty:
                raise RuntimeError("PySR search completed but found no equations.")

            best_sympy_expr = model.sympy()
            expression_str = str(best_sympy_expr)

            predictions = model.predict(X)

            return {
                "expression": expression_str,
                "predictions": predictions.tolist(),
                "details": {}
            }

        except Exception:
            # If PySR fails, fall back to a simple linear regression model.
            x1, x2, x3, x4 = X[:, 0], X[:, 1], X[:, 2], X[:, 3]
            A = np.column_stack([x1, x2, x3, x4, np.ones_like(x1)])

            try:
                coeffs, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
                a, b, c, d, e = coeffs
                expression = f"{a:.8f}*x1 + {b:.8f}*x2 + {c:.8f}*x3 + {d:.8f}*x4 + {e:.8f}"
                predictions = a * x1 + b * x2 + c * x3 + d * x4 + e
            except np.linalg.LinAlgError:
                # If least squares fails, return a constant prediction.
                expression = f"{np.mean(y):.8f}"
                predictions = np.full_like(y, np.mean(y))

            return {
                "expression": expression,
                "predictions": predictions.tolist(),
                "details": {}
            }