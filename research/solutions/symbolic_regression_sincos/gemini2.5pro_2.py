import numpy as np
from pysr import PySRRegressor
import sympy

class Solution:
    def __init__(self, **kwargs):
        """
        Constructor for the Solution class.
        """
        pass

    def solve(self, X: np.ndarray, y: np.ndarray) -> dict:
        """
        Finds a symbolic expression for the given dataset.

        This method uses PySR (Python Symbolic Regression) to search for a
        closed-form expression that fits the data. The configuration is
        tailored to the "SinCos" dataset, prioritizing trigonometric
        functions and leveraging the available CPU cores for an efficient search.

        Args:
            X: Feature matrix of shape (n, 2) with columns 'x1', 'x2'.
            y: Target values of shape (n,).

        Returns:
            A dictionary containing the best-found symbolic expression,
            optional predictions, and details like complexity.
        """
        # Configure PySRRegressor based on problem hints and environment
        model = PySRRegressor(
            niterations=50,             # Iterations for the search; balanced for speed and accuracy
            populations=24,             # Number of parallel populations (3 per core)
            population_size=50,         # Size of each population
            procs=8,                    # Utilize all 8 available vCPUs
            binary_operators=["+", "-", "*"],  # Basic arithmetic operations
            unary_operators=["sin", "cos"],    # Focus on trigonometric functions as hinted by dataset name
            maxsize=20,                 # Limit expression complexity to prevent overfitting
            model_selection="best",     # Select the equation with the best score (accuracy vs. complexity)
            verbosity=0,                # Suppress console output
            progress=False,             # Disable progress bar
            random_state=42,            # For reproducibility
            temp_equation_file=True,    # Save progress to a file for robustness
        )

        model.fit(X, y, variable_names=["x1", "x2"])

        # Fallback mechanism in case PySR fails to find any equations
        if not hasattr(model, 'equations_') or model.equations_.empty:
            # Fit a simple linear model as a robust baseline
            x1_col, x2_col = X[:, 0], X[:, 1]
            A = np.column_stack([x1_col, x2_col, np.ones_like(x1_col)])
            try:
                coeffs, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
                a, b, c = coeffs
                expression = f"{a:.6f}*x1 + {b:.6f}*x2 + {c:.6f}"
            except np.linalg.LinAlgError:
                # If linear regression fails, use the mean as a constant expression
                expression = f"{np.mean(y):.6f}"

            return {
                "expression": expression,
                "predictions": None,  # Evaluator will compute predictions
                "details": {"fallback": "linear_model"}
            }

        # Extract results from the best model found by PySR
        best_sympy_expr = model.sympy()
        expression_str = str(best_sympy_expr)
        
        # Generate predictions from the best model
        predictions = model.predict(X)
        
        # Extract the complexity score calculated by PySR
        complexity = int(model.equations_.iloc[-1]['complexity'])

        return {
            "expression": expression_str,
            "predictions": predictions.tolist(),
            "details": {"complexity": complexity}
        }