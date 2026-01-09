import numpy as np
from pysr import PySRRegressor
import sympy
import warnings

# Suppress common PySR warnings for cleaner output
warnings.filterwarnings("ignore", category=FutureWarning, module='pysr.*')
warnings.filterwarnings("ignore", category=UserWarning, module='pysr.*')
warnings.filterwarnings("ignore", category=RuntimeWarning)


class Solution:
    """
    A solution for the Symbolic Regression Benchmark on the McCormick Dataset.

    This solution uses the PySR (Python Symbolic Regression) library to discover
    the underlying mathematical expression from the given data. The configuration
    of the PySRRegressor is tuned based on the known properties of the McCormick
    function, which involves trigonometric and polynomial terms.
    """

    def __init__(self, **kwargs):
        """
        Initializes the Solution class.
        """
        # The true McCormick function is sin(x1 + x2) + (x1 - x2)**2 - 1.5*x1 + 2.5*x2 + 1.
        # This hints at the necessary operators and functions.
        self.model = PySRRegressor(
            # Search configuration
            niterations=60,         # Number of generations for the search
            populations=24,         # Number of populations to evolve in parallel (3 per core)
            population_size=50,     # Number of expressions per population

            # Operators to build expressions
            # 'pow' is used for the '**' operator
            binary_operators=["+", "-", "*", "pow"],
            unary_operators=["sin", "cos"], # sin is in the true function, cos is a good alternative

            # Constraints to guide the search and reduce the search space
            # The (x1-x2)**2 term suggests a power of 2 is important.
            constraints={'pow': 2},
            # Prevent nesting of trig functions to avoid overly complex solutions
            nested_constraints={
                "sin": {"sin": 0, "cos": 0},
                "cos": {"sin": 0, "cos": 0}
            },

            # Complexity control
            maxsize=30,             # Maximum complexity (number of nodes) of an expression
            annealing=True,         # Use simulated annealing to favor simpler expressions
            
            # Performance and environment settings
            procs=8,                # Utilize all 8 vCPUs available in the environment
            
            # Reproducibility and output control
            random_state=42,        # For deterministic results
            verbosity=0,            # Suppress verbose output
            progress=False,         # Disable progress bar
            
            # Use PySR's default model selection which balances accuracy and complexity
            model_selection="best"
        )

    def solve(self, X: np.ndarray, y: np.ndarray) -> dict:
        """
        Fits a symbolic expression to the data using PySR.

        Args:
            X: Feature matrix of shape (n, 2)
            y: Target values of shape (n,)

        Returns:
            A dictionary containing the discovered expression, predictions, and details.
        """
        # Fit the PySR model to the data
        self.model.fit(X, y, variable_names=["x1", "x2"])

        # Check if the search was successful
        if not hasattr(self.model, 'equations_') or self.model.equations_.shape[0] == 0:
            # Fallback to a simple constant if PySR fails to find any expressions
            expression = "0.0"
            predictions = np.zeros_like(y)
            complexity = 0
        else:
            # Retrieve the best expression found by PySR
            # .sympy() returns the best model's expression as a sympy object
            best_sympy_expr = self.model.sympy()
            
            # Convert the sympy expression to a standard Python-evaluable string
            # sympy automatically formats Pow(x, 2) as x**2
            expression = str(best_sympy_expr)
            
            # Generate predictions using the best model
            predictions = self.model.predict(X)
            
            # Get the complexity of the best model from the results dataframe
            best_equation_details = self.model.get_best()
            complexity = best_equation_details.complexity

        return {
            "expression": expression,
            "predictions": predictions.tolist(),
            "details": {"complexity": int(complexity)}
        }