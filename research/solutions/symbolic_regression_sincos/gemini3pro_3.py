import numpy as np
import pandas as pd
import sympy
from pysr import PySRRegressor
import warnings
import os

# Suppress warnings and output to keep the execution clean
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class Solution:
    def __init__(self, **kwargs):
        pass

    def solve(self, X: np.ndarray, y: np.ndarray) -> dict:
        """
        Solves the symbolic regression problem using PySRRegressor.
        
        Args:
            X: Feature matrix of shape (n, 2)
            y: Target values of shape (n,)
            
        Returns:
            dict with keys: expression, predictions, details
        """
        # Variable names required by the problem statement
        variable_names = ["x1", "x2"]
        
        # PySRRegressor configuration optimized for the environment (8 vCPUs)
        # Using a balance of population size and iterations to fit within compute constraints
        # while searching the space of trigonometric functions effectively.
        model = PySRRegressor(
            niterations=50,
            binary_operators=["+", "-", "*", "/"],
            unary_operators=["sin", "cos", "exp", "log"],
            populations=16,           # 2 populations per core (8 vCPUs)
            population_size=40,
            maxsize=30,               # Limit complexity
            ncycles_per_iteration=500,
            parsimony=0.001,          # Penalize complexity to find simpler solutions
            verbosity=0,              # Suppress output
            progress=False,
            random_state=42,
            procs=8,                  # Utilize all 8 vCPUs
            multithreading=False,     # Use multiprocessing
            temp_equation_file=True,
            delete_tempfiles=True,
            model_selection="best",   # Select best model based on score/complexity
        )

        try:
            # Fit the symbolic regression model
            model.fit(X, y, variable_names=variable_names)

            # Retrieve the best expression found
            # model.sympy() returns a sympy object of the best equation
            best_expr = model.sympy()
            expression = str(best_expr)

            # Generate predictions using the fitted model
            predictions = model.predict(X)
            
            # Ensure predictions are a list/array
            if hasattr(predictions, "tolist"):
                predictions = predictions.tolist()

        except Exception:
            # Fallback to Linear Regression if PySR fails (e.g., timeout, convergence issue)
            # This ensures the solution always returns a valid result.
            
            # Prepare data for linear regression (add intercept)
            X_aug = np.column_stack([X, np.ones(X.shape[0])])
            
            # Least squares fit
            coeffs, _, _, _ = np.linalg.lstsq(X_aug, y, rcond=None)
            
            # Construct linear expression string
            # coeffs order: x1, x2, intercept
            w1, w2, b = coeffs
            expression = f"({w1} * x1) + ({w2} * x2) + {b}"
            
            # Calculate predictions
            predictions = (X_aug @ coeffs).tolist()

        return {
            "expression": expression,
            "predictions": predictions,
            "details": {}
        }