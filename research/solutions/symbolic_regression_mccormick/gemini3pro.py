import numpy as np
from pysr import PySRRegressor
import sympy

class Solution:
    def __init__(self, **kwargs):
        pass

    def solve(self, X: np.ndarray, y: np.ndarray) -> dict:
        """
        Solves the symbolic regression problem using PySR.
        Target: McCormick function structure (trig + polynomial).
        """
        
        # Configure PySRRegressor
        # We include operators relevant to the McCormick function (trig, poly)
        # ^ is included for compact representation of power terms like (x-y)^2
        model = PySRRegressor(
            niterations=50,
            binary_operators=["+", "-", "*", "/", "^"],
            unary_operators=["sin", "cos"],
            populations=20,
            population_size=40,
            maxsize=35,
            model_selection="best",  # Selects model with best score/complexity trade-off
            verbosity=0,
            progress=False,
            random_state=42,
            procs=8,
            multiprocessing=True,
            deterministic=True,
            tempdir=None,
        )

        # Fit the model to the data
        # Variable names must match the problem specification
        model.fit(X, y, variable_names=["x1", "x2"])

        # Retrieve the best expression found as a sympy object
        best_expr = model.sympy()

        # Convert to Python-evaluable string (SymPy uses ** for power, which is valid)
        expression = str(best_expr)

        # Generate predictions using the fitted model
        predictions = model.predict(X)

        return {
            "expression": expression,
            "predictions": predictions.tolist(),
            "details": {}
        }