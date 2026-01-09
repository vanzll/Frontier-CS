import numpy as np
import pandas as pd
import sympy
from pysr import PySRRegressor

class Solution:
    def __init__(self, **kwargs):
        pass

    def solve(self, X: np.ndarray, y: np.ndarray) -> dict:
        """
        Fits a symbolic regression model to the peaks dataset using PySR.
        """
        # Configure PySRRegressor
        # We include '^' in binary_operators to efficiently capture polynomial terms (mapped to ** in output).
        # We include 'exp' as it is critical for the peaks function.
        # We use 'best' model selection to balance complexity and accuracy.
        model = PySRRegressor(
            niterations=200,
            timeout_in_seconds=180,
            binary_operators=["+", "-", "*", "/", "^"],
            unary_operators=["exp", "sin", "cos", "log"],
            populations=20,
            population_size=50,
            maxsize=50,
            ncycles_per_iteration=500,
            verbosity=0,
            progress=False,
            random_state=42,
            procs=8,
            multithreading=False,
            model_selection="best",
            loss="L2DistLoss",
            turbo=True
        )

        # Fit the model
        # Explicitly name variables x1, x2 to match the requirements
        model.fit(X, y, variable_names=["x1", "x2"])

        try:
            # Retrieve the best expression as a SymPy object
            # PySR handles the conversion of internal operators (like ^) to SymPy equivalents (Pow)
            best_expr = model.sympy()
            
            # Convert SymPy expression to a valid Python string
            # SymPy automatically formats powers as '**', which is allowed
            expression = str(best_expr)
            
            # Generate predictions using the fitted model
            predictions = model.predict(X)
            
            # Ensure predictions are in list format
            if isinstance(predictions, np.ndarray):
                predictions = predictions.tolist()

        except Exception:
            # Fallback baseline in case of failure
            expression = "0 * x1"
            predictions = [0.0] * len(y)

        return {
            "expression": expression,
            "predictions": predictions,
            "details": {}
        }