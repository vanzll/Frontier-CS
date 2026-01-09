import numpy as np
import sympy
from pysr import PySRRegressor

class Solution:
    def __init__(self, **kwargs):
        pass

    def solve(self, X: np.ndarray, y: np.ndarray) -> dict:
        """
        Solves the symbolic regression problem using PySR.
        Target: McCormick function derived dataset.
        """
        # Configure PySRRegressor
        # The McCormick function is f(x,y) = sin(x+y) + (x-y)^2 - 1.5x + 2.5y + 1
        # It involves sin, +, -, * (for squares). 
        # We optimize parameters for 8 vCPUs and the known function characteristics.
        model = PySRRegressor(
            niterations=60,            # Sufficient iterations for convergence
            binary_operators=["+", "-", "*", "/"],
            unary_operators=["sin", "cos"], # exp and log are not needed for McCormick
            populations=16,            # Parallel populations
            population_size=40,
            maxsize=35,                # Allow enough complexity for polynomial expansion
            verbosity=0,
            progress=False,
            random_state=42,
            deterministic=True,
            model_selection="best",    # optimize for lowest error
        )

        # Fit the model
        # variable_names must be x1, x2 to match output requirements
        model.fit(X, y, variable_names=["x1", "x2"])

        try:
            # specific processing to ensure valid python string
            best_expr = model.sympy()
            expression = str(best_expr)
            predictions = model.predict(X)
        except Exception:
            # Fallback to linear regression
            x1, x2 = X[:, 0], X[:, 1]
            A = np.column_stack([x1, x2, np.ones_like(x1)])
            coeffs, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
            a, b, c = coeffs
            expression = f"{a}*x1 + {b}*x2 + {c}"
            predictions = A @ coeffs

        return {
            "expression": expression,
            "predictions": predictions.tolist(),
            "details": {}
        }