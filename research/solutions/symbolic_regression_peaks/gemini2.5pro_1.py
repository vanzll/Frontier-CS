import numpy as np
from pysr import PySRRegressor

class Solution:
    def __init__(self, **kwargs):
        """
        Initializes the Solution class by configuring a PySRRegressor instance.
        The parameters are tuned to find a potentially complex function like "peaks"
        within a reasonable time on a multi-core CPU environment.
        """
        self.model = PySRRegressor(
            # Search configuration
            niterations=80,
            populations=32,
            population_size=40,

            # Complexity settings
            maxsize=35,
            parsimony=0.001,

            # Operators and functions to use in expressions
            binary_operators=["+", "-", "*", "/"],
            unary_operators=["exp", "cos", "sin", "log", "square"],

            # Custom operator handling and search constraints
            extra_sympy_mappings={"square": lambda x: x**2},
            nested_constraints={"exp": {"exp": 0}, "log": {"log": 0}},

            # Model selection criteria
            model_selection="best",

            # Environment and performance settings
            procs_per_population=1,
            temp_equation_file=True,
            
            # Logging and reproducibility
            progress=False,
            verbosity=0,
            random_state=42,
        )

    def solve(self, X: np.ndarray, y: np.ndarray) -> dict:
        """
        Fits the PySR model to the data and returns the best symbolic expression.

        Args:
            X: Feature matrix of shape (n, 2)
            y: Target values of shape (n,)

        Returns:
            A dictionary containing the symbolic expression, predictions, and details.
        """
        # Fit the symbolic regression model with specified variable names
        self.model.fit(X, y, variable_names=["x1", "x2"])

        # Fallback if PySR finds no equations
        if not hasattr(self.model, 'equations_') or self.model.equations_.shape[0] == 0:
            return self._get_fallback_solution(X, y)

        # Extract the best equation found
        best_equation = self.model.equations_.iloc[-1]
        pysr_complexity = int(best_equation['complexity'])
        
        # Convert the best symbolic expression to a string
        best_sympy_expr = self.model.sympy()
        expression_str = str(best_sympy_expr)
        
        # Generate predictions using the best-found model
        predictions = self.model.predict(X)

        return {
            "expression": expression_str,
            "predictions": predictions.tolist(),
            "details": {"complexity": pysr_complexity}
        }

    def _get_fallback_solution(self, X: np.ndarray, y: np.ndarray) -> dict:
        """
        Provides a linear regression fallback solution if PySR fails.
        """
        x1, x2 = X[:, 0], X[:, 1]
        A = np.column_stack([x1, x2, np.ones_like(x1)])
        
        try:
            coeffs, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
            a, b, c = coeffs
            expression = f"{a:.6f}*x1 + {b:.6f}*x2 + {c:.6f}"
            predictions = a * x1 + b * x2 + c
            details = {"complexity": 5}
        except np.linalg.LinAlgError:
            expression = "0.0"
            predictions = np.zeros_like(y)
            details = {"complexity": 1}
            
        return {
            "expression": expression,
            "predictions": predictions.tolist(),
            "details": details
        }