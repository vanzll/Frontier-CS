import numpy as np
import pandas as pd
import sympy
from pysr import PySRRegressor

class Solution:
    def __init__(self, **kwargs):
        pass

    def solve(self, X: np.ndarray, y: np.ndarray) -> dict:
        """
        Args:
            X: Feature matrix of shape (n, 4)
            y: Target values of shape (n,)

        Returns:
            dict with keys "expression", "predictions", "details"
        """
        # PySR configuration optimized for 8 vCPUs and mixed complexity
        # "Mixed PolyExp" suggests polynomials and exponentials are key.
        # "Gaussian-like damping" strongly suggests exp(-x^2) terms.
        model = PySRRegressor(
            niterations=200,            # Higher iterations for 4D convergence
            binary_operators=["+", "-", "*", "/"],
            unary_operators=["exp", "sin", "cos", "log"],
            populations=16,             # 2 populations per CPU (8 CPUs)
            population_size=50,
            maxsize=40,                 # Allow reasonably complex expressions
            verbosity=0,
            progress=False,
            model_selection="best",     # Select best model based on score (accuracy/complexity)
            random_state=42,
            procs=8,                    # Use all available CPUs
            multithreading=False,       # Multiprocessing is generally more robust for PySR
            temp_equation_file=True,
            turbo=True
        )

        variable_names = ["x1", "x2", "x3", "x4"]
        
        try:
            # Fit the symbolic regression model
            model.fit(X, y, variable_names=variable_names)
            
            # Retrieve the best expression found
            # model.sympy() returns a sympy object; str() converts it to a Python-compatible string
            best_expr = model.sympy()
            expression = str(best_expr)
            
            # Generate predictions using the fitted model
            predictions = model.predict(X)
            
            # Handle potential NaN in predictions if the expression is invalid for some inputs
            if np.any(np.isnan(predictions)):
                raise ValueError("NaN predictions from symbolic model")

        except Exception:
            # Fallback to Linear Regression if PySR fails or produces invalid results
            # This ensures we always return a valid response
            x1, x2, x3, x4 = X[:, 0], X[:, 1], X[:, 2], X[:, 3]
            # Design matrix with intercept
            A = np.column_stack([x1, x2, x3, x4, np.ones_like(x1)])
            coeffs, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
            
            # Construct linear expression string
            c = coeffs
            expression = f"({c[0]})*x1 + ({c[1]})*x2 + ({c[2]})*x3 + ({c[3]})*x4 + ({c[4]})"
            predictions = A @ coeffs

        return {
            "expression": expression,
            "predictions": predictions.tolist(),
            "details": {}
        }