import numpy as np
import tempfile
import os
from pysr import PySRRegressor

class Solution:
    def __init__(self, **kwargs):
        pass

    def solve(self, X: np.ndarray, y: np.ndarray) -> dict:
        """
        Solves the symbolic regression problem using PySR.
        
        Args:
            X: Feature matrix of shape (n, 2)
            y: Target values of shape (n,)
            
        Returns:
            dict containing the best expression string and predictions.
        """
        # Create a temporary directory for PySR's equation file
        # to ensure clean execution and avoid file system clutter.
        with tempfile.TemporaryDirectory() as tmpdir:
            equation_file = os.path.join(tmpdir, "hall_of_fame.csv")
            
            # Initialize PySRRegressor
            # Optimized for 8 vCPUs and the Ripple dataset characteristics
            model = PySRRegressor(
                niterations=1000,             # Large iteration budget, limited by time
                binary_operators=["+", "-", "*", "/", "**"],
                unary_operators=["sin", "cos", "exp", "log"],
                populations=24,               # Parallelism: 3 * 8 vCPUs
                population_size=40,
                maxsize=50,                   # Allow complexity for ripple modulation terms
                timeout_in_seconds=280,       # ~4.5 minutes execution limit
                procs=8,                      # Use all 8 available cores
                multiprocessing=True,
                verbosity=0,
                progress=False,
                random_state=42,
                equation_file=equation_file,
                loss="mse",
                model_selection="best",       # Choose best based on accuracy-complexity trade-off
            )

            try:
                # Fit the model to the data
                # variable_names ensures the output expression uses x1, x2
                model.fit(X, y, variable_names=["x1", "x2"])

                # Retrieve the best expression as a sympy object
                best_expr = model.sympy()
                
                # Convert to valid Python expression string
                expression_str = str(best_expr)
                
                # Generate predictions
                predictions = model.predict(X)
                
                return {
                    "expression": expression_str,
                    "predictions": predictions.tolist(),
                    "details": {}
                }

            except Exception:
                # Fallback: Linear Regression if PySR fails or times out without result
                # Fits a plane: y = a*x1 + b*x2 + c
                x1, x2 = X[:, 0], X[:, 1]
                A = np.column_stack([x1, x2, np.ones_like(x1)])
                
                # Compute coefficients via least squares
                coeffs, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
                a, b, c = coeffs
                
                expression_str = f"({a})*x1 + ({b})*x2 + ({c})"
                predictions = A @ coeffs
                
                return {
                    "expression": expression_str,
                    "predictions": predictions.tolist(),
                    "details": {"fallback": True}
                }