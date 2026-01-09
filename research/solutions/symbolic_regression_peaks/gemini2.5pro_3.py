import numpy as np
from pysr import PySRRegressor

class Solution:
    def __init__(self, **kwargs):
        """
        Initializes the Solution object.
        """
        pass

    def solve(self, X: np.ndarray, y: np.ndarray) -> dict:
        """
        Finds a symbolic expression for the given data using PySR.

        Args:
            X: Feature matrix of shape (n, 2).
            y: Target values of shape (n,).

        Returns:
            A dictionary containing the symbolic expression, predictions,
            and details like complexity.
        """
        # The evaluation environment provides 8 vCPUs.
        N_PROCS = 8

        # Configure PySRRegressor with parameters tuned for a thorough search
        # within a reasonable time limit on the given hardware.
        model = PySRRegressor(
            # Search budget
            niterations=200,
            populations=32,
            population_size=50,
            
            # Operators allowed by the problem specification
            binary_operators=["+", "-", "*", "/", "**"],
            unary_operators=["sin", "cos", "exp", "log"],
            
            # Complexity constraints
            maxsize=40,
            
            # Parallelization settings to match the environment
            procs=N_PROCS,
            
            # For reproducibility and clean output
            random_state=42,
            verbosity=0,
            progress=False,
            
            # Efficiency: stop the search if it's no longer improving
            early_stop_condition="stop_if_no_progress_over_x_generations(25)",
            
            # Use default L2 loss (MSE), which aligns with the scoring metric
            loss="L2DistLoss()",
        )

        # Run the symbolic regression search
        model.fit(X, y, variable_names=["x1", "x2"])

        # Check if any valid equation was found
        if len(model.equations_) == 0:
            # Fallback to a simple linear model if PySR fails to find a solution
            x1, x2 = X[:, 0], X[:, 1]
            A = np.column_stack([x1, x2, np.ones_like(x1)])
            try:
                coeffs, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
                a, b, c = coeffs
                expression = f"{a:.8f}*x1 + {b:.8f}*x2 + {c:.8f}"
                predictions = a * x1 + b * x2 + c
                # Complexity: 2 mult, 2 add -> 4 binary ops. C = 2 * 4 = 8
                details = {"complexity": 8}
            except np.linalg.LinAlgError:
                # Absolute fallback in case of singular matrix
                expression = "0.0"
                predictions = np.zeros_like(y)
                details = {"complexity": 0}
            
            return {
                "expression": expression,
                "predictions": predictions.tolist(),
                "details": details
            }

        # Retrieve the best-performing equation found by PySR
        best_equation = model.get_best()
        
        # Convert the sympy expression to a standard Python-evaluable string
        expression = str(best_equation.sympy_format)
        
        # PySR's predict() method automatically uses the best model found
        predictions = model.predict(X)
        
        # Extract the complexity of the best model
        complexity = best_equation.complexity

        return {
            "expression": expression,
            "predictions": predictions.tolist(),
            "details": {"complexity": int(complexity)}
        }