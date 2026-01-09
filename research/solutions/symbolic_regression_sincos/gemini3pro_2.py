import numpy as np
from pysr import PySRRegressor
import sympy
import warnings

# Suppress unnecessary warnings
warnings.filterwarnings("ignore")

class Solution:
    def __init__(self, **kwargs):
        pass

    def solve(self, X: np.ndarray, y: np.ndarray) -> dict:
        """
        Fits a symbolic regression model to finding the closed-form expression f(x1, x2) -> y.
        """
        # Configure PySRRegressor
        # Using settings optimized for 8 vCPUs (populations ~ 3*cores)
        # Including trigonometric operators as per "SinCos" dataset hint
        model = PySRRegressor(
            niterations=100,            # Sufficient iterations for convergence
            binary_operators=["+", "-", "*", "/", "^"],
            unary_operators=["sin", "cos", "exp", "log"],
            populations=24,             # Parallelism: 3 * 8 vCPUs
            population_size=40,
            maxsize=40,                 # Allow reasonably complex expressions
            model_selection="best",     # Optimize for accuracy/complexity trade-off
            verbosity=0,
            progress=False,
            random_state=42,
            deterministic=True,
            timeout_in_seconds=300,     # Time limit safety
            temp_equation_file=True,
            delete_tempfiles=True
        )

        # Fit the symbolic regression model
        # variable_names map input columns to x1, x2 matching output requirement
        model.fit(X, y, variable_names=["x1", "x2"])

        # Retrieve the best expression found
        try:
            # model.sympy() returns the SymPy representation of the best equation
            best_equation = model.sympy()
            expression = str(best_equation)
            predictions = model.predict(X)
        except Exception:
            # Fallback to linear regression if symbolic regression fails for any reason
            # (e.g., timeout before any equation found)
            A = np.column_stack([X, np.ones(X.shape[0])])
            coeffs, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
            a, b, c = coeffs[0], coeffs[1], coeffs[2]
            expression = f"{a}*x1 + {b}*x2 + {c}"
            predictions = X @ coeffs[:2] + c

        return {
            "expression": expression,
            "predictions": predictions,
            "details": {}
        }