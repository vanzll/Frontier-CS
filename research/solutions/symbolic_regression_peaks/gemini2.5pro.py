import numpy as np
from pysr import PySRRegressor
import sympy

class Solution:
    def __init__(self, **kwargs):
        """
        Initialize the solution by configuring the PySR regressor.
        The parameters are tuned for a complex 'peaks'-like function,
        leveraging the multi-core environment.
        """
        self.model = PySRRegressor(
            # Search effort parameters
            niterations=100,
            populations=24,
            population_size=50,

            # Operators: including power `**` for polynomial components
            binary_operators=["+", "-", "*", "/", "**"],
            unary_operators=["exp", "cos", "sin", "log"],

            # Complexity settings
            maxsize=35,
            parsimony=0.001,

            # Constraints to improve stability and guide the search
            nested_constraints={
                "exp": {"exp": 0},
                "log": {"log": 0},
            },
            
            # Environment settings
            procs=8,
            temp_equation_file=True,
            
            # Reproducibility and output control
            random_state=42,
            verbosity=0,
            progress=False,
        )

    def solve(self, X: np.ndarray, y: np.ndarray) -> dict:
        """
        Find a symbolic expression for the given data using PySR.
        
        Args:
            X: Feature matrix of shape (n, 2)
            y: Target values of shape (n,)

        Returns:
            A dictionary containing the symbolic expression, predictions, and details.
        """
        
        self.model.fit(X, y, variable_names=["x1", "x2"])

        # Check if the model found any equations
        if not hasattr(self.model, 'equations_') or self.model.equations_.empty:
            # Fallback if PySR fails
            expression = "0.0"
            predictions = np.zeros_like(y)
            details = {"complexity": 0}
        else:
            # Get the best equation's properties
            best_equation = self.model.get_best()
            
            # PySRRegressor.sympy() and .predict() automatically use the best model
            expression = str(self.model.sympy())
            predictions = self.model.predict(X)
            complexity = best_equation['complexity']
            
            details = {"complexity": int(complexity)}

        return {
            "expression": expression,
            "predictions": predictions.tolist(),
            "details": details,
        }