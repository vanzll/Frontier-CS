import numpy as np
from pysr import PySRRegressor
import sympy as sp

class Solution:
    """
    An expert programmer's solution to the Symbolic Regression Benchmark.
    This implementation uses the PySR library to discover a symbolic
    expression that fits the provided data. The hyperparameters for
    PySR are tuned to provide a good balance between search time and
    solution quality on a CPU-only environment.
    """
    def __init__(self, **kwargs):
        """
        The constructor for the Solution class.
        It is intentionally left empty as per the problem's API specification.
        """
        pass

    def solve(self, X: np.ndarray, y: np.ndarray) -> dict:
        """
        Learns a symbolic expression from the input data.

        Args:
            X: Feature matrix of shape (n, 2) with columns 'x1', 'x2'.
            y: Target values of shape (n,).

        Returns:
            A dictionary containing the learned symbolic expression.
        """
        # Configure the PySR regressor.
        # The parameters are chosen to be effective for the given problem
        # and computational constraints.
        model = PySRRegressor(
            # Search parameters
            niterations=100,
            populations=32,
            population_size=50,
            
            # Operators to build expressions from. Based on problem spec and
            # knowledge of the underlying McCormick function.
            binary_operators=["+", "-", "*", "/", "pow"],
            unary_operators=["sin", "cos"],
            
            # Complexity control and scoring
            maxsize=30,
            parsimony=0.005,  # Encourages simpler expressions
            
            # Efficiency and environment settings
            procs=0,  # Use all available CPU cores
            random_state=42, # For reproducibility
            verbosity=0,
            progress=False,
            early_stop_condition="f(loss, complexity) = loss < 1e-6",
            temp_equation_file=True,
        )

        try:
            # Fit the model to the data
            model.fit(X, y, variable_names=["x1", "x2"])
            
            # Check if any valid equations were discovered
            if model.equations_ is None or model.equations_.shape[0] == 0:
                raise RuntimeError("PySR did not find any equations.")

            # Retrieve the best expression as a sympy object and convert to string
            best_expr_sympy = model.sympy()
            expression = str(best_expr_sympy)

        except (Exception, RuntimeError):
            # Fallback to a robust linear model if PySR fails for any reason
            x1, x2 = X[:, 0], X[:, 1]
            A = np.column_stack([x1, x2, np.ones_like(x1)])
            try:
                coeffs, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
                a, b, c = coeffs
                expression = f"{a:.8f}*x1 + {b:.8f}*x2 + {c:.8f}"
            except np.linalg.LinAlgError:
                # If least squares fails, fallback to the mean
                expression = f"{np.mean(y):.8f}"
            
        return {
            "expression": expression,
            # Let the evaluator compute predictions from the expression
            "predictions": None,
            "details": {}
        }