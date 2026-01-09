import numpy as np
from pysr import PySRRegressor
import sympy
import warnings

# Suppress common warnings from PySR and its dependencies for cleaner output
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

class Solution:
    """
    A solution for the Symbolic Regression Benchmark - Mixed PolyExp 4D Dataset.
    This implementation uses the PySR library, a state-of-the-art tool for symbolic
    regression based on regularized evolution.
    """
    def __init__(self, **kwargs):
        """
        The constructor is a no-op, as the model is instantiated and used
        entirely within the `solve` method.
        """
        pass

    def solve(self, X: np.ndarray, y: np.ndarray) -> dict:
        """
        Learns a symbolic expression to predict y from X.

        The method configures and runs a PySRRegressor, which is well-suited
        for the multi-core CPU environment. Parameters are chosen to balance
        search time with the potential complexity of a 4-dimensional problem.

        If PySR fails or finds no suitable expressions, it gracefully falls back
        to a simple linear regression model to ensure a robust solution.

        Args:
            X: Feature matrix of shape (n, 4) with columns x1, x2, x3, x4.
            y: Target values of shape (n,).

        Returns:
            A dictionary containing the learned expression, predictions, and
            optional details like complexity.
        """
        variable_names = ["x1", "x2", "x3", "x4"]
        
        try:
            # Configure PySR for the 4D problem, balancing search time and solution quality.
            model = PySRRegressor(
                niterations=75,         # A reasonable number of generations for a benchmark
                populations=32,         # Good diversity for 8 cores
                population_size=50,     # Standard population size
                maxsize=35,             # Allow moderately complex expressions
                
                # Operators allowed in the search space
                binary_operators=["+", "-", "*", "/"],
                unary_operators=["exp", "cos", "sin", "log"],
                
                # Environment-specific settings
                procs=8,                # Utilize all available CPU cores
                
                # Performance and stability settings
                batching=True,          # Use mini-batches for speed on large datasets
                batch_size=1024,        # A larger batch size can be more stable
                
                # Settings for reproducibility and clean execution
                random_state=42,
                verbosity=0,
                progress=False,
                
                # Default model selection criteria balances accuracy and complexity
                model_selection="best",
            )
            
            # Run the symbolic regression search
            model.fit(X, y, variable_names=variable_names)

            # Check for a valid result
            if not hasattr(model, 'equations_') or model.equations_.empty:
                raise ValueError("PySR did not find any equations.")

            # Extract the best solution
            best_sympy_expr = model.sympy()
            expression_str = str(best_sympy_expr)
            predictions = model.predict(X)
            
            # The last equation in the DataFrame is the one selected as 'best'
            complexity = int(model.equations_.iloc[-1]['complexity'])

            return {
                "expression": expression_str,
                "predictions": predictions.tolist(),
                "details": {"complexity": complexity}
            }

        except (Exception, SystemExit):
            # Fallback to a simple linear regression model if PySR fails
            x1, x2, x3, x4 = X[:, 0], X[:, 1], X[:, 2], X[:, 3]
            A = np.column_stack([x1, x2, x3, x4, np.ones_like(x1)])
            
            try:
                coeffs, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
                c1, c2, c3, c4, c_intercept = coeffs
            except np.linalg.LinAlgError:
                # Handle cases where least squares fails
                c1, c2, c3, c4 = 0.0, 0.0, 0.0, 0.0
                c_intercept = np.mean(y)

            expression = (
                f"{c1:.8f}*x1 + {c2:.8f}*x2 + {c3:.8f}*x3 + "
                f"{c4:.8f}*x4 + {c_intercept:.8f}"
            )
            predictions = c1 * x1 + c2 * x2 + c3 * x3 + c4 * x4 + c_intercept

            return {
                "expression": expression,
                "predictions": predictions.tolist(),
                "details": {}
            }