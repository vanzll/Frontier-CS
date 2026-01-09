import numpy as np
from pysr import PySRRegressor

class Solution:
    def __init__(self, **kwargs):
        pass

    def solve(self, X: np.ndarray, y: np.ndarray) -> dict:
        """
        Finds a symbolic expression for the given data using PySR.

        Args:
            X: Feature matrix of shape (n, 2)
            y: Target values of shape (n,)

        Returns:
            A dictionary with the symbolic expression, predictions, and details.
        """
        # Configure PySRRegressor with parameters suited for the environment and problem.
        # The parameters are chosen to balance search time and solution quality.
        # Increased populations and population_size enhance search diversity.
        # A slightly lower maxsize encourages simpler, more generalizable expressions.
        model = PySRRegressor(
            niterations=50,
            populations=20,
            population_size=50,
            binary_operators=["+", "*", "-", "/"],
            unary_operators=["sin", "cos", "exp", "log"],
            maxsize=20,
            random_state=42,
            verbosity=0,
            progress=False,
            # Let PySR automatically manage processes based on available cores.
        )

        # Fit the model to the data, specifying variable names for the output expression.
        model.fit(X, y, variable_names=["x1", "x2"])

        # Handle the case where PySR fails to find any valid expression.
        if not hasattr(model, 'equations_') or model.equations_.empty:
            expression = "0.0"
            predictions = np.zeros_like(y)
            complexity = 0
        else:
            # Extract the best solution found by PySR.
            best_equation = model.equations_.iloc[-1]
            
            # Get the best expression as a sympy object.
            sympy_expr = model.sympy()
            
            # Convert the sympy expression to a string.
            expression = str(sympy_expr)
            
            # Generate predictions from the best model.
            predictions = model.predict(X)
            
            # Get the complexity value from PySR's output.
            complexity = int(best_equation['complexity'])

        return {
            "expression": expression,
            "predictions": predictions.tolist(),
            "details": {"complexity": complexity}
        }