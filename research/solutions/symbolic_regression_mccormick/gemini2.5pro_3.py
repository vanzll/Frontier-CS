import numpy as np
from pysr import PySRRegressor
import sympy as sp
import warnings

# Suppress common warnings from dependencies for a cleaner output.
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


class Solution:
    """
    Implements a solution for the Symbolic Regression Benchmark on the McCormick dataset.
    This solution utilizes the PySR library to perform symbolic regression and find a
    closed-form mathematical expression that models the provided data.
    """

    def __init__(self, **kwargs):
        """
        Initializes the Solution class. No specific setup is required here.
        """
        pass

    def solve(self, X: np.ndarray, y: np.ndarray) -> dict:
        """
        Finds a symbolic expression f(x1, x2) that fits the given data (X, y).

        The method configures and runs a PySRRegressor, a powerful tool for
        symbolic regression that uses genetic programming to discover mathematical
        formulas. Parameters are chosen based on the problem's nature (McCormick
        function) and the provided computational environment.

        Args:
            X: A numpy.ndarray of shape (n, 2) containing feature values for x1 and x2.
            y: A numpy.ndarray of shape (n,) containing the target values.

        Returns:
            A dictionary with the following keys:
            - "expression": A string representing the discovered Python-evaluable
                            mathematical expression.
            - "predictions": Set to None, as the evaluation environment will compute
                             predictions from the expression.
            - "details": An empty dictionary, as no extra details are provided.
        """

        # Configure PySRRegressor. The McCormick function's structure
        # (sin(x1+x2) + (x1-x2)**2 - 1.5*x1 + 2.5*x2 + 1) informs the choice
        # of operators to focus the search.
        model = PySRRegressor(
            # Search parameters optimized for the 8-vCPU environment.
            niterations=60,
            populations=40,
            population_size=50,

            # Define the building blocks for the expressions.
            binary_operators=["+", "-", "*", "pow"],
            unary_operators=["sin", "cos"],

            # Set constraints to guide the search towards simpler, more plausible expressions.
            maxsize=30,  # Limit expression complexity.
            # Disallow nesting of trigonometric functions (e.g., sin(cos(x))).
            nested_constraints={"sin": {"sin": 0, "cos": 0}, "cos": {"sin": 0, "cos": 0}},

            # Execution settings.
            procs=8,
            timeout_in_seconds=300,  # A safety net to ensure timely completion.

            # Settings for reproducibility and clean output.
            random_state=42,
            verbosity=0,
            progress=False,
        )

        try:
            # Run the symbolic regression search.
            model.fit(X, y, variable_names=["x1", "x2"])

            # If the search completes but finds no valid equations, it will raise an error
            # when trying to access the results. This is caught by the except block.
            if len(model.equations_) == 0:
                raise IndexError("PySR search finished without finding any equations.")

            # Retrieve the best-scoring expression as a SymPy object.
            sympy_expr = model.sympy()

            # Use SymPy's simplification capabilities to reduce the expression's complexity,
            # which can improve the final score.
            simplified_expr = sp.simplify(sympy_expr)

            # Convert the final SymPy expression to a string that can be evaluated by Python.
            expression = str(simplified_expr)

        except (IndexError, AttributeError, RuntimeError):
            # If PySR fails for any reason (e.g., timeout, no equations found),
            # fall back to a robust linear regression model as a baseline.
            x1, x2 = X[:, 0], X[:, 1]
            A = np.column_stack([x1, x2, np.ones_like(x1)])
            try:
                # Fit coefficients for the model: a*x1 + b*x2 + c
                coeffs, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
                a, b, c = coeffs
                expression = f"{a:.6f}*x1 + {b:.6f}*x2 + {c:.6f}"
            except np.linalg.LinAlgError:
                # As a last resort if least squares fails, return the mean.
                expression = f"{np.mean(y):.6f}"

        return {
            "expression": expression,
            "predictions": None,
            "details": {},
        }