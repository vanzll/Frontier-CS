import numpy as np
import pandas as pd
from pysr import PySRRegressor
import sympy

class Solution:
    def __init__(self, **kwargs):
        pass

    def solve(self, X: np.ndarray, y: np.ndarray) -> dict:
        """
        Solves the symbolic regression problem using PySR.
        """
        # Configure PySRRegressor
        # Using settings optimized for the 8 vCPU environment
        model = PySRRegressor(
            niterations=100,
            binary_operators=["+", "-", "*", "/"],
            unary_operators=["sin", "cos", "exp", "log"],
            populations=16,
            population_size=40,
            maxsize=35,
            verbosity=0,
            progress=False,
            random_state=42,
            procs=8,
            multiprocessing=True,
            temp_equation_file=True,
            delete_tempfiles=True,
            model_selection="best",
        )

        try:
            # Fit the model
            model.fit(X, y, variable_names=["x1", "x2"])
            
            # Retrieve the best expression found
            best_sympy = model.sympy()
            expression = str(best_sympy)
            
            # Generate predictions
            predictions = model.predict(X)
            
            # Check for numerical instability (NaNs/Infs)
            if not np.all(np.isfinite(predictions)):
                raise ValueError("Model predictions contain NaN or Inf")
                
        except Exception:
            # Fallback strategy: Linear Regression
            try:
                x1, x2 = X[:, 0], X[:, 1]
                A = np.column_stack([x1, x2, np.ones_like(x1)])
                coeffs, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
                a, b, c = coeffs
                
                expression = f"{a}*x1 + {b}*x2 + {c}"
                predictions = a * x1 + b * x2 + c
            except Exception:
                # Ultimate fallback: Mean predictor
                mean_val = float(np.mean(y))
                expression = str(mean_val)
                predictions = np.full(y.shape, mean_val)

        return {
            "expression": expression,
            "predictions": predictions.tolist(),
            "details": {}
        }