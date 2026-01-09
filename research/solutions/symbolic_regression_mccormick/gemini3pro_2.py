import numpy as np
import pandas as pd
import sympy as sp
from pysr import PySRRegressor
import warnings

class Solution:
    def __init__(self, **kwargs):
        pass

    def solve(self, X: np.ndarray, y: np.ndarray) -> dict:
        """
        Solves the symbolic regression problem for the McCormick function dataset.
        Target function structure: sin(x1 + x2) + (x1 - x2)^2 - 1.5*x1 + 2.5*x2 + 1
        """
        # Suppress warnings to keep output clean
        warnings.filterwarnings("ignore")
        
        # Configure PySRRegressor
        # Optimized for the available 8 vCPUs
        model = PySRRegressor(
            niterations=100,  # Sufficient iterations for convergence on this complexity
            binary_operators=["+", "-", "*", "/", "^"],  # Include ^ to easily find squared terms
            unary_operators=["sin", "cos"],  # sin is essential for McCormick
            maxsize=40,  # Allow enough complexity for the full expression
            populations=20,
            population_size=40,
            model_selection="best",  # Select best model based on score (accuracy vs complexity)
            loss="loss(prediction, target) = (prediction - target)^2",  # MSE
            random_state=42,
            deterministic=True,
            procs=8,  # Utilize all 8 vCPUs
            verbosity=0,
            progress=False,
            timeout_in_seconds=300,  # Safety timeout
            temp_equation_file=True,  # Avoid cluttering with csv files
        )
        
        # Fit the model
        # PySR requires X to be (n_samples, n_features)
        # Variable names x1, x2 match the problem statement
        model.fit(X, y, variable_names=["x1", "x2"])
        
        # Retrieve the best expression found
        try:
            best_expr = model.sympy()
            expression = str(best_expr)
        except Exception:
            # Fallback if model fails to return a valid expression
            expression = "x1 + x2"
            
        # Generate predictions using the fitted model
        try:
            predictions = model.predict(X).tolist()
        except Exception:
            predictions = None
            
        return {
            "expression": expression,
            "predictions": predictions,
            "details": {}
        }