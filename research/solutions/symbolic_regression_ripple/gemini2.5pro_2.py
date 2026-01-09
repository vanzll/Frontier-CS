import numpy as np
from pysr import PySRRegressor
import warnings

# Suppress PySR warnings for a cleaner output, as it can be verbose.
warnings.filterwarnings("ignore", category=FutureWarning, module="pysr.*")
warnings.filterwarnings("ignore", category=UserWarning, module="pysr.*")

class Solution:
    """
    An expert programmer's solution for the Symbolic Regression Benchmark.
    This implementation uses the PySR library, a state-of-the-art tool for
    symbolic regression based on regularized evolution.
    """

    def __init__(self, **kwargs):
        """
        The constructor for the Solution class.
        """
        pass

    def solve(self, X: np.ndarray, y: np.ndarray) -> dict:
        """
        Finds a symbolic expression that fits the given data.

        Args:
            X: A numpy array of shape (n, 2) representing the features (x1, x2).
            y: A numpy array of shape (n,) representing the target values.

        Returns:
            A dictionary containing the discovered symbolic expression,
            optional predictions, and optional details like complexity.
        """

        # Configure the PySR regressor with parameters optimized for performance
        # and search effectiveness within the given CPU-only environment.
        model = PySRRegressor(
            # Search parameters tailored for a thorough search
            niterations=100,           # More generations to find better expressions
            populations=48,            # Increased populations for diversity (6 per core)
            population_size=50,        # Larger population size for more expressions
            maxsize=30,                # Allow for reasonably complex expressions

            # Environment and execution control
            procs=8,                   # Use all available vCPUs for parallel computation
            timeout_in_seconds=290,    # A safety timeout to ensure completion

            # Define the building blocks for expressions as per problem spec
            binary_operators=["+", "-", "*", "/", "pow"], # 'pow' for the '**' operator
            unary_operators=["cos", "sin", "exp", "log"],

            # Technical settings for robustness and precision
            elementwise_functions={"log": "log_abs"}, # Use a safe log for non-positive inputs
            extra_precision=64,        # Enhance precision of discovered constants
            optimizer_iterations=30,   # More optimization steps for constants

            # Add constraints to guide the search towards simpler, more stable models
            nested_constraints={
                "sin": 1, "cos": 1, "exp": 1, "log": 1,
                "pow": 1,  # Disallow deeply nested powers like (x^y)^z
            },

            # Ensure deterministic results
            random_state=42,

            # Suppress console output for clean execution
            verbosity=0,
            progress=False,
        )

        try:
            # Run the symbolic regression search
            model.fit(X, y, variable_names=["x1", "x2"])

            # Check if the search yielded any valid equations
            if not hasattr(model, 'equations_') or model.equations_.empty:
                return self._fallback_solve(X, y)

            # Retrieve the best expression found by PySR.
            # model.sympy() returns the best model's expression in SymPy format.
            expression_sympy = model.sympy()
            expression_str = str(expression_sympy)

            # Generate predictions using the final, best-fit model
            predictions = model.predict(X)

            # Extract the complexity of the best model for the optional details dictionary.
            # PySR's complexity is the number of nodes in the expression tree.
            complexity = model.equations_.iloc[-1]['complexity']

            return {
                "expression": expression_str,
                "predictions": predictions.tolist(),
                "details": {"complexity": int(complexity)}
            }

        except Exception:
            # If PySR fails for any reason (e.g., timeout, internal error),
            # return a robust baseline solution.
            return self._fallback_solve(X, y)

    def _fallback_solve(self, X: np.ndarray, y: np.ndarray) -> dict:
        """
        Provides a simple linear regression model as a fallback.
        This ensures a valid solution is always returned.
        """
        x1, x2 = X[:, 0], X[:, 1]

        # Design matrix for y = a*x1 + b*x2 + c
        A = np.c_[x1, x2, np.ones_like(x1)]

        try:
            # Use least squares to find the best-fit coefficients [a, b, c]
            coeffs, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
            a, b, c = coeffs
        except np.linalg.LinAlgError:
            # If least squares fails (e.g., singular matrix), use a very safe default
            a, b, c = 0.0, 0.0, np.mean(y)

        # Format the linear expression with the found coefficients
        expression = f"{a:.6f}*x1 + {b:.6f}*x2 + {c:.6f}"

        # Calculate predictions from this fallback model
        predictions = a * x1 + b * x2 + c

        return {
            "expression": expression,
            "predictions": predictions.tolist(),
            "details": {}
        }