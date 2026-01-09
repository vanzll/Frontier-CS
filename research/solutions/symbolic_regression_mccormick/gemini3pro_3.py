import numpy as np
import pandas as pd
import sympy as sp
from pysr import PySRRegressor
import os
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

class Solution:
    def __init__(self, **kwargs):
        pass

    def solve(self, X: np.ndarray, y: np.ndarray) -> dict:
        # Define operators based on the McCormick function characteristics (trig + polynomial)
        # Limiting operators improves search efficiency
        binary_operators = ["+", "-", "*"]
        unary_operators = ["sin", "cos"]
        
        # Configure PySRRegressor
        # Optimized for 8 vCPUs and CPU-only environment
        model = PySRRegressor(
            niterations=60,              # Sufficient iterations for convergence
            binary_operators=binary_operators,
            unary_operators=unary_operators,
            populations=16,              # 2x vCPUs for effective parallelism
            population_size=40,
            maxsize=40,                  # Allow enough complexity for polynomial expansion
            model_selection="best",      # Balance accuracy and complexity
            loss="mse",
            verbosity=0,
            progress=False,
            random_state=42,
            procs=8,                     # Use all available vCPUs
            multiprocessing=True,
            temp_equation_file=True,     # Use temp directory for intermediate files
            early_stop_condition="f(loss, complexity) = (loss < 1e-6) && (complexity < 40)",
        )

        # Fit the model
        # variable_names must match the required output format (x1, x2)
        model.fit(X, y, variable_names=["x1", "x2"])

        try:
            # Retrieve the best expression as a sympy object
            best_expr = model.sympy()
            
            # Convert to string
            expression = str(best_expr)
            
            # Generate predictions
            predictions = model.predict(X)
            
            # Ensure predictions are a list
            if isinstance(predictions, np.ndarray):
                predictions = predictions.tolist()
            else:
                predictions = list(predictions)
                
        except Exception:
            # Fallback strategy if symbolic regression fails
            # Linear baseline
            expression = "x1 + x2"
            predictions = (X[:, 0] + X[:, 1]).tolist()

        return {
            "expression": expression,
            "predictions": predictions,
            "details": {}
        }