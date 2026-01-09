import numpy as np
import warnings
from pysr import PySRRegressor

# Suppress warnings for cleaner execution
warnings.filterwarnings("ignore")

class Solution:
    def __init__(self, **kwargs):
        pass

    def solve(self, X: np.ndarray, y: np.ndarray) -> dict:
        """
        Solves the Symbolic Regression problem for the Peaks dataset using PySR.
        """
        # Configure PySRRegressor with settings optimized for the Peaks function
        # Peaks function features exponential and polynomial interactions.
        # 8 vCPUs allow for parallel populations.
        model = PySRRegressor(
            niterations=100,             # Sufficient iterations for convergence
            binary_operators=["+", "-", "*", "/"],
            unary_operators=["exp", "sin", "cos", "log"],
            populations=16,              # 2 populations per vCPU
            population_size=40,          # Robust population size
            maxsize=50,                  # Allow for the complexity of the Peaks function
            model_selection="best",      # Choose the best model on the pareto frontier
            verbosity=0,                 # Suppress output
            progress=False,              # Suppress progress bar
            random_state=42,             # Reproducibility
            timeout_in_seconds=300,      # 5-minute timeout safeguard
            turbo=True,                  # Enable high-performance mode
            tempdir=None,                # Use default temporary directory
        )

        # Fit the model to the data
        # Explicitly name variables x1, x2 as required by the output format
        model.fit(X, y, variable_names=["x1", "x2"])

        try:
            # Retrieve the best expression found as a SymPy object
            best_expr_sympy = model.sympy()
            
            # Convert SymPy expression to a Python string
            # SymPy uses '**' for power, which is compatible with the requirements
            expression = str(best_expr_sympy)
            
            # Compute predictions using the fitted model
            predictions = model.predict(X)
            
        except Exception:
            # Fallback: Simple Linear Regression if symbolic regression fails
            # This ensures the method always returns a valid result matching the API
            x1, x2 = X[:, 0], X[:, 1]
            A = np.column_stack([x1, x2, np.ones_like(x1)])
            coeffs, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
            a, b, c = coeffs
            
            expression = f"{a}*x1 + {b}*x2 + {c}"
            predictions = a * x1 + b * x2 + c

        # Construct the result dictionary
        return {
            "expression": expression,
            "predictions": predictions.tolist(),
            "details": {}
        }