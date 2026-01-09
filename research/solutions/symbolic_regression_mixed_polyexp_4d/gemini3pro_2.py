import numpy as np
import pandas as pd
import sympy
from pysr import PySRRegressor
import warnings

# Suppress warnings for cleaner execution
warnings.filterwarnings("ignore")

class Solution:
    def __init__(self, **kwargs):
        pass

    def solve(self, X: np.ndarray, y: np.ndarray) -> dict:
        """
        Solves the symbolic regression problem for 4D Mixed PolyExp data.
        Uses PySRRegressor with parameters optimized for the 8 vCPU environment.
        """
        variable_names = ["x1", "x2", "x3", "x4"]
        
        try:
            # Configure PySRRegressor
            # - niterations/populations optimized for 8 cores
            # - operators set to match problem description and allowed list
            # - binary_operators includes "^" to efficiently capture polynomial terms
            model = PySRRegressor(
                niterations=50,
                populations=24,           # 3 populations per core (8 cores)
                population_size=40,
                procs=8,
                multithreading=False,     # Use multiprocessing backend
                binary_operators=["+", "-", "*", "/", "^"],
                unary_operators=["exp", "sin", "cos", "log"],
                maxsize=35,               # Allow sufficient complexity for 4D interactions
                model_selection="best",   # Select best model based on accuracy/complexity trade-off
                verbosity=0,
                progress=False,
                temp_equation_file=True,
                delete_tempfiles=True,
                random_state=42,
                deterministic=True,
                timeout_in_seconds=300,   # Safety timeout
            )
            
            # Fit model to data
            model.fit(X, y, variable_names=variable_names)
            
            # Retrieve the best expression as a SymPy object
            # PySR handles the conversion from internal representation to SymPy
            best_sympy = model.sympy()
            
            # Convert SymPy expression to Python string
            # SymPy automatically converts Powers (^) to Python syntax (**)
            expression = str(best_sympy)
            
            # Generate predictions using the fitted model
            predictions = model.predict(X)
            
            # Validation check
            if expression == "nan" or expression is None:
                raise ValueError("Invalid expression generated")
                
            return {
                "expression": expression,
                "predictions": predictions.tolist(),
                "details": {}
            }
            
        except Exception as e:
            # Fallback: Linear Regression
            # Used if PySR fails to run, converge, or generates invalid output
            return self._fallback_linear_regression(X, y, variable_names)

    def _fallback_linear_regression(self, X, y, variable_names):
        # Implement robust least squares linear regression
        n_samples = X.shape[0]
        # Add bias term column
        X_aug = np.column_stack([np.ones(n_samples), X])
        
        # Solve weights
        beta, _, _, _ = np.linalg.lstsq(X_aug, y, rcond=None)
        
        # Construct expression string: y = b0 + b1*x1 + ...
        # Using parentheses ensures correct operator precedence
        expr_parts = [str(beta[0])]
        for i, name in enumerate(variable_names):
            coef = beta[i+1]
            expr_parts.append(f"+ ({coef} * {name})")
            
        expression = "".join(expr_parts)
        predictions = X_aug @ beta
        
        return {
            "expression": expression,
            "predictions": predictions.tolist(),
            "details": {"fallback": True}
        }