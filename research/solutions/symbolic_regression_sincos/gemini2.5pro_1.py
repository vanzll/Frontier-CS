import numpy as np
from pysr import PySRRegressor
import sympy

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
            A dictionary containing the symbolic expression and predictions.
        """
        # Configure PySR for a focused search on trigonometric functions,
        # which is hinted at by the dataset name "SinCos".
        # Parameters are set for a reasonably intensive search on an 8-core CPU.
        model = PySRRegressor(
            niterations=50,
            populations=24,
            population_size=40,
            maxsize=30,
            binary_operators=["+", "-", "*", "/"],
            unary_operators=["sin", "cos"],
            model_selection="best",
            procs=8,
            random_state=42,
            verbosity=0,
            progress=False,
            temp_equation_file=True,
            deterministic=True,
        )

        # Fit the model to the data
        model.fit(X, y, variable_names=["x1", "x2"])

        # Process the results
        if not hasattr(model, 'equations_') or model.equations_.empty:
            # Fallback to a simple constant if PySR fails to find an expression
            expression = "0.0"
            predictions = np.zeros_like(y)
        else:
            # Retrieve the best symbolic expression
            best_expr_sympy = model.sympy()
            
            # Evaluate symbolic constants like 'pi' to their numeric values
            # to ensure the final string is a valid Python expression.
            # Using a precision of 15 decimal places.
            evaluated_expr = best_expr_sympy.evalf(15)
            expression = str(evaluated_expr)
            
            # Generate predictions using the best-found model
            predictions = model.predict(X)

        return {
            "expression": expression,
            "predictions": predictions.tolist(),
            "details": {}
        }