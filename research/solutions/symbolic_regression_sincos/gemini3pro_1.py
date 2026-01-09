import numpy as np
import pandas as pd
import sympy
from pysr import PySRRegressor
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

class Solution:
    def __init__(self, **kwargs):
        pass

    def solve(self, X: np.ndarray, y: np.ndarray) -> dict:
        """
        Args:
            X: Feature matrix of shape (n, 2)
            y: Target values of shape (n,)

        Returns:
            dict containing "expression" and "predictions"
        """
        
        # Fallback method: Linear Regression
        def linear_fallback():
            x1_f, x2_f = X[:, 0], X[:, 1]
            A = np.column_stack([x1_f, x2_f, np.ones_like(x1_f)])
            coeffs, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
            a, b, c = coeffs
            # Format expression
            expr_str = f"({a})*x1 + ({b})*x2 + ({c})"
            preds = a * x1_f + b * x2_f + c
            return expr_str, preds

        try:
            # Configure PySRRegressor
            # Optimized for 8 vCPUs (procs=8) and SinCos dataset characteristics
            model = PySRRegressor(
                niterations=50,
                binary_operators=["+", "-", "*", "/"],
                unary_operators=["sin", "cos", "exp", "log"],
                populations=16,     # 2 populations per vCPU
                population_size=40,
                maxsize=30,         # Sufficient for trigonometric combinations
                verbosity=0,
                progress=False,
                random_state=42,
                procs=8,
                multiprocessing=True,
                # Additional safety settings
                turbo=True
            )

            # Fit the model to the data
            # Explicitly providing variable names as required by the output format
            model.fit(X, y, variable_names=["x1", "x2"])

            # Retrieve the best expression found (highest score)
            # model.sympy() returns a sympy object, convert to string
            best_expr = model.sympy()
            expression = str(best_expr)
            
            # Generate predictions using the fitted model
            predictions = model.predict(X)

        except Exception as e:
            # In case of any failure in PySR (e.g. timeout, backend issue), use fallback
            expression, predictions = linear_fallback()

        return {
            "expression": expression,
            "predictions": predictions.tolist(),
            "details": {}
        }