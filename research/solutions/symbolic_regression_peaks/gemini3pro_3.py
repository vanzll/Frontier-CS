import numpy as np
from pysr import PySRRegressor
import sympy

class Solution:
    def __init__(self, **kwargs):
        pass

    def solve(self, X: np.ndarray, y: np.ndarray) -> dict:
        # Configure PySR for the Peaks dataset
        # We use a robust configuration with timeout to respect environment limits
        model = PySRRegressor(
            niterations=100,
            binary_operators=["+", "-", "*", "/", "^"],
            unary_operators=["exp", "log", "sin", "cos"],
            populations=8,  # Parallelize across available vCPUs
            population_size=40,
            maxsize=45,     # Allow sufficient complexity for peaks function
            timeout_in_seconds=180, # Set a hard time limit
            verbosity=0,
            progress=False,
            random_state=42,
            procs=8,
            model_selection="best",
            constraints={
                "^": (-1, 5), # Limit power exponents to reasonable range
            },
            nested_constraints={
                "exp": {"exp": 0, "log": 0, "sin": 0, "cos": 0}, # Prevent deeply nested transcendentals
                "log": {"exp": 0, "log": 0, "sin": 0, "cos": 0},
                "sin": {"sin": 0, "cos": 0},
                "cos": {"sin": 0, "cos": 0},
            },
            temp_equation_file=True,
            delete_tempfiles=True
        )

        # Fit the model
        # variable_names ensures the output expression uses x1, x2
        model.fit(X, y, variable_names=["x1", "x2"])

        try:
            # Retrieve the best expression found
            # model.sympy() returns a sympy object; str() converts it to Python-compatible string
            best_expr = model.sympy()
            expression = str(best_expr)
            predictions = model.predict(X)
        except Exception:
            # Fallback to linear regression if symbolic regression fails
            x1, x2 = X[:, 0], X[:, 1]
            A = np.column_stack([x1, x2, np.ones_like(x1)])
            coeffs, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
            a, b, c = coeffs
            expression = f"{a}*x1 + {b}*x2 + {c}"
            predictions = a * x1 + b * x2 + c

        return {
            "expression": expression,
            "predictions": predictions.tolist(),
            "details": {}
        }