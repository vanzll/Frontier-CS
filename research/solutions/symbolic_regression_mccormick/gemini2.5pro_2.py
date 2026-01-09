import numpy as np
from pysr import PySRRegressor
import sympy

class Solution:
    def __init__(self, **kwargs):
        """
        Initializes the Solution object.
        Any keyword arguments are ignored.
        """
        pass

    def solve(self, X: np.ndarray, y: np.ndarray) -> dict:
        """
        Finds a symbolic expression for the given data using PySR.

        Args:
            X: Feature matrix of shape (n, 2)
            y: Target values of shape (n,)

        Returns:
            A dictionary containing the symbolic expression, predictions,
            and details like complexity.
        """
        # Configure PySRRegressor to best suit the problem and environment.
        # The parameters are tuned for a balance of search thoroughness and speed
        # on an 8-core CPU environment.
        model = PySRRegressor(
            # --- Search Configuration ---
            niterations=60,
            populations=32,
            population_size=35,
            
            # --- Operators ---
            # All allowed operators are provided to the search algorithm.
            binary_operators=["+", "-", "*", "/", "**"],
            unary_operators=["sin", "cos", "exp", "log"],

            # --- Complexity Control ---
            # Maxsize is set slightly above the known complexity of the true
            # function to allow for variations.
            maxsize=25,

            # Key optimization: Align PySR's internal complexity metric with
            # the competition's scoring metric: C = 2*(#binary) + 1*(#unary).
            # This guides the evolutionary search towards expressions that
            # are likely to score higher.
            complexity_of_operators={
                "+": 2, "-": 2, "*": 2, "/": 2, "**": 2,
                "sin": 1, "cos": 1, "exp": 1, "log": 1,
            },

            # --- Environment/Execution ---
            procs=8,
            verbosity=0,
            progress=False,
            random_state=42,
        )

        # Fit the model to the data.
        # `variable_names` ensures the output expression uses 'x1' and 'x2'.
        model.fit(X, y, variable_names=["x1", "x2"])

        # Retrieve the best equation based on the score (accuracy vs. complexity).
        # This will raise an error if no equations were found, which is an
        # acceptable failure mode.
        best_equation = model.get_best()
        
        # Get the best expression in SymPy format.
        sympy_expr = model.sympy()
        
        # Convert the SymPy expression to a Python-evaluable string.
        expression_str = str(sympy_expr)

        # Generate predictions from the final expression.
        predictions = model.predict(X)

        # Extract the complexity, which was calculated using our custom metric.
        complexity = best_equation.complexity

        # Return the results in the specified dictionary format.
        return {
            "expression": expression_str,
            "predictions": predictions.tolist(),
            "details": {"complexity": int(complexity)}
        }