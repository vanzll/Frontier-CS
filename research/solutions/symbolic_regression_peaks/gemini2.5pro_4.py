import numpy as np
from pysr import PySRRegressor
import sympy

class Solution:
    def __init__(self, **kwargs):
        pass

    def solve(self, X: np.ndarray, y: np.ndarray) -> dict:
        """
        Finds a symbolic expression that fits the given data using PySR.
        
        Args:
            X: Feature matrix of shape (n, 2)
            y: Target values of shape (n,)

        Returns:
            A dictionary with the symbolic expression, predictions, and details.
        """
        # The "Peaks" dataset is characterized by exponential and polynomial terms.
        # The PySRRegressor is configured to search for expressions with these components.
        # Hyperparameters are set for a thorough search within a typical benchmark time limit.
        model = PySRRegressor(
            # Search configuration:
            niterations=60,       # Number of generations for evolution
            populations=16,       # Parallel populations (2 * num_cores is a good rule of thumb)
            population_size=40,   # Number of expressions in each population

            # Resource configuration:
            procs=8,              # Use all available vCPUs for parallel processing

            # Operators to build expressions:
            # "pow" for polynomial terms and "exp" for the characteristic peaks.
            binary_operators=["+", "-", "*", "/", "pow"],
            unary_operators=["exp", "cos", "sin"],

            # Complexity control:
            maxsize=30,           # Maximum number of operators/variables in an expression

            # For reproducibility and clean execution:
            random_state=42,
            verbosity=0,
            progress=False,
            temp_equation_file=True, # Manages temporary files created during the run
        )

        model.fit(X, y, variable_names=["x1", "x2"])

        # After fitting, check if any valid equations were found.
        if not hasattr(model, 'equations_') or model.equations_.empty:
            # Provide a fallback solution if PySR fails to find any expressions.
            expression = "0.0"
            predictions = np.zeros_like(y)
            complexity = 0
        else:
            # PySR automatically selects the best equation based on a score that
            # balances accuracy (MSE) and complexity.
            # The .predict() and .sympy() methods default to this best equation.
            
            predictions = model.predict(X)
            sympy_expr = model.sympy()
            expression = str(sympy_expr)
            
            # Extract the complexity of the best equation for scoring purposes.
            best_idx = model.equations_.score.idxmax()
            complexity = int(model.equations_.loc[best_idx, "complexity"])

        return {
            "expression": expression,
            "predictions": predictions.tolist(),
            "details": {
                "complexity": complexity
            }
        }