import numpy as np
import pandas as pd
from pysr import PySRRegressor
import sympy
import warnings
import os

# Suppress warnings for cleaner execution
warnings.filterwarnings("ignore")

class Solution:
    def __init__(self, **kwargs):
        pass

    def solve(self, X: np.ndarray, y: np.ndarray) -> dict:
        """
        Solves the Mixed PolyExp 4D symbolic regression problem.
        Uses PySR with parameters optimized for an 8-core CPU environment.
        """
        
        # Configure PySR Regressor
        # - timeout_in_seconds=300: limit to 5 minutes to respect evaluation constraints
        # - procs=8: utilize all 8 vCPUs
        # - nested_constraints: restrict search space to likely physical forms (PolyExp)
        # - binary/unary operators: cover required operations including exp/log/trig
        model = PySRRegressor(
            niterations=10000,  # High count, effectively limited by timeout
            binary_operators=["+", "-", "*", "/"],
            unary_operators=["sin", "cos", "exp", "log"],
            maxsize=50,  # Allow for sufficient complexity for 4D interactions
            populations=24,  # ~3 populations per core
            population_size=50,
            ncycles_per_iteration=500,
            model_selection="best",  # Optimizes trade-off between score and complexity
            loss="loss(x, y) = (x - y)^2",  # MSE
            random_state=42,
            procs=8,
            multithreading=False,  # Multiprocessing is preferred
            verbosity=0,
            progress=False,
            timeout_in_seconds=300,
            temp_equation_file=True,
            delete_tempfiles=True,
            nested_constraints={
                "sin": {"sin": 0, "cos": 0, "exp": 0, "log": 0},
                "cos": {"sin": 0, "cos": 0, "exp": 0, "log": 0},
                "exp": {"exp": 0, "log": 1, "sin": 1, "cos": 1},
                "log": {"exp": 0, "log": 0, "sin": 0, "cos": 0}
            },
            # Prevent extremely complex constants
            complexity_of_constants=2
        )
        
        # Fit the model
        # variable_names ensures the output string uses x1, x2, x3, x4
        model.fit(X, y, variable_names=["x1", "x2", "x3", "x4"])
        
        try:
            # Retrieve the best expression found
            best_expr = model.sympy()
            expression = str(best_expr)
            
            # Generate predictions
            predictions_raw = model.predict(X)
            
            # Ensure predictions are in list format
            if isinstance(predictions_raw, np.ndarray):
                predictions = predictions_raw.tolist()
            elif hasattr(predictions_raw, "values"):
                predictions = predictions_raw.values.tolist()
            else:
                predictions = list(predictions_raw)
                
        except Exception:
            # Fallback to Linear Regression if symbolic search fails or returns invalid result
            try:
                # Add bias term
                X_aug = np.column_stack([X, np.ones(X.shape[0])])
                coeffs, _, _, _ = np.linalg.lstsq(X_aug, y, rcond=None)
                
                # Construct Python-evaluable linear expression
                terms = []
                for i in range(4):
                    terms.append(f"({coeffs[i]}*x{i+1})")
                terms.append(f"{coeffs[4]}")
                expression = " + ".join(terms)
                
                predictions = (X_aug @ coeffs).tolist()
            except Exception:
                # Ultimate fallback
                expression = "0"
                predictions = [0.0] * len(y)

        return {
            "expression": expression,
            "predictions": predictions,
            "details": {}
        }