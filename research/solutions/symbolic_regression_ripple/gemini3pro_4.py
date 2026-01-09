import numpy as np
import pandas as pd
import sympy
from pysr import PySRRegressor
import warnings

# Suppress warnings to keep output clean
warnings.filterwarnings("ignore")

class Solution:
    def __init__(self, **kwargs):
        pass

    def solve(self, X: np.ndarray, y: np.ndarray) -> dict:
        """
        Solves the symbolic regression problem using PySR.
        """
        # Subsample data if it's too large to ensure we finish within time limits
        # while maintaining enough data for accurate regression.
        n_samples = X.shape[0]
        if n_samples > 3000:
            rng = np.random.RandomState(42)
            indices = rng.choice(n_samples, 3000, replace=False)
            X_train = X[indices]
            y_train = y[indices]
        else:
            X_train = X
            y_train = y

        # Configure PySRRegressor
        # Optimized for 8 vCPUs and the "Ripple" dataset characteristics
        # (likely polynomial amplitude with trigonometric components)
        model = PySRRegressor(
            niterations=60,                   # Iterations to run
            binary_operators=["+", "-", "*", "/"],
            unary_operators=["sin", "cos", "exp", "log"],
            populations=16,                   # Parallel populations (2x cores)
            population_size=40,               # Individuals per population
            ncycles_per_iteration=500,        # Evolution steps per iteration
            maxsize=45,                       # Allow complex expressions
            verbosity=0,                      # Suppress output
            progress=False,
            procs=8,                          # Use all 8 vCPUs
            multiprocessing=True,
            random_state=42,
            model_selection="best",           # Choose best score (balance accuracy/complexity)
            parsimony=0.001,                  # Complexity penalty
            temp_equation_file=True,          # Use temp files
            delete_tempfiles=True,
            timeout_in_seconds=300            # Safety timeout (5 minutes)
        )

        try:
            # Fit the model
            model.fit(X_train, y_train, variable_names=["x1", "x2"])
            
            # Extract the best symbolic expression
            best_expr = model.sympy()
            expression = str(best_expr)
            
            # Generate predictions on the full dataset using the discovered model
            predictions = model.predict(X)
            
        except Exception:
            # Fallback if symbolic regression fails
            expression = "0"
            predictions = np.zeros(len(y))

        return {
            "expression": expression,
            "predictions": predictions.tolist(),
            "details": {}
        }