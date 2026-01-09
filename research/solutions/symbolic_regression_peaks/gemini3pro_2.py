import numpy as np
from pysr import PySRRegressor
import sympy
import warnings

# Suppress warnings to ensure clean output
warnings.filterwarnings("ignore")

class Solution:
    def __init__(self, **kwargs):
        pass

    def solve(self, X: np.ndarray, y: np.ndarray) -> dict:
        """
        Fits a symbolic regression model to the data (x1, x2) -> y.
        """
        # Configure PySRRegressor
        # The Peaks function involves exponentials and polynomials.
        # We include "^" in binary_operators to efficiently capture polynomial terms
        # (reduced complexity compared to repeated multiplication).
        model = PySRRegressor(
            niterations=100,  # Sufficient iterations for convergence
            binary_operators=["+", "-", "*", "/", "^"],
            unary_operators=["exp", "sin", "cos", "log"],
            populations=20,
            population_size=40,
            maxsize=50,  # Allow enough complexity for the Peaks function
            
            # Constraints to guide the search and prevent invalid nesting
            nested_constraints={
                "exp": {"exp": 0, "log": 0, "sin": 1, "cos": 1},
                "log": {"exp": 0, "log": 0},
                "sin": {"sin": 0, "cos": 0},
                "cos": {"sin": 0, "cos": 0}
            },
            
            # Execution parameters
            verbosity=0,
            progress=False,
            random_state=42,
            procs=8,
            multiprocessing=True,
            
            # Model selection
            model_selection="best",
            loss="L2DistLoss",
            timeout_in_seconds=300, # 5 minute timeout safety
            
            # File handling
            temp_equation_file=True,
            delete_tempfiles=True
        )
        
        # Fit the model
        # Variable names must be set to x1, x2 to match the required output expression format
        model.fit(X, y, variable_names=["x1", "x2"])
        
        # Extract the best symbolic expression
        try:
            # model.sympy() returns a SymPy object. 
            # str() converts it to a valid Python expression (e.g., using '**' for power).
            best_expr = model.sympy()
            expression = str(best_expr)
        except Exception:
            # Fallback
            expression = "0.0"
            
        # Generate predictions
        try:
            predictions = model.predict(X)
        except Exception:
            predictions = np.zeros(len(y))

        return {
            "expression": expression,
            "predictions": predictions.tolist(),
            "details": {}
        }