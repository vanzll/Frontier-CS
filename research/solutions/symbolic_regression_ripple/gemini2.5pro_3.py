import numpy as np
from pysr import PySRRegressor
import sympy

class Solution:
    def __init__(self, **kwargs):
        """
        Initialize the PySRRegressor model.
        Parameters are tuned for performance on an 8-core CPU environment and
        the characteristics of the ripple dataset.
        """
        self.model = PySRRegressor(
            niterations=60,
            populations=24,
            population_size=50,
            binary_operators=["+", "-", "*", "/"],
            unary_operators=["sin", "cos", "exp", "log"],
            maxsize=35,
            procs=8,
            random_state=42,
            verbosity=0,
            progress=False,
            # Constraints to prevent redundant compositions of trigonometric functions
            nested_constraints={
                "sin": {"sin": 0, "cos": 0},
                "cos": {"sin": 0, "cos": 0},
            },
            # Use default optimizer settings for constant discovery
        )

    def solve(self, X: np.ndarray, y: np.ndarray) -> dict:
        """
        Finds a symbolic expression for the given data using PySR.

        The strategy involves feature engineering to guide the symbolic regression
        process. Based on the "ripple" nature of the data, a radial feature
        `r_sq = x1**2 + x2**2` is introduced. This simplifies the search space,
        allowing PySR to find the underlying structure more efficiently.

        Args:
            X: Feature matrix of shape (n, 2) with columns 'x1', 'x2'.
            y: Target values of shape (n,).

        Returns:
            A dictionary containing the symbolic expression, predictions, and details.
        """
        # 1. Feature Engineering: Create a radial feature `r_sq = x1^2 + x2^2`
        # This is based on the insight that a "ripple" pattern is often radially symmetric.
        x1 = X[:, 0]
        x2 = X[:, 1]
        r_sq = x1**2 + x2**2
        
        X_transformed = np.c_[X, r_sq]
        variable_names = ["x1", "x2", "r_sq"]

        # 2. Fit the PySR model on the transformed data
        try:
            self.model.fit(X_transformed, y, variable_names=variable_names)
        except Exception:
            # Fallback in case of an unexpected error during fitting
            expression = "0.0"
            return {
                "expression": expression,
                "predictions": np.zeros_like(y).tolist(),
                "details": {}
            }
        
        # 3. Handle cases where no solution is found
        if not hasattr(self.model, 'equations_') or self.model.equations_.empty:
            expression = "0.0"
            return {
                "expression": expression,
                "predictions": np.zeros_like(y).tolist(),
                "details": {"complexity": 0}
            }

        # 4. Retrieve the best expression and post-process it
        best_expr_with_r_sq = self.model.sympy()

        # Define sympy symbols to perform substitution
        x1_var, x2_var, r_sq_var = sympy.symbols('x1 x2 r_sq')
        
        # Define the substitution rule for our engineered feature
        r_sq_sub_expr = x1_var**2 + x2_var**2

        # Substitute `r_sq` back into the expression to get the final form
        final_sympy_expr = best_expr_with_r_sq.subs(r_sq_var, r_sq_sub_expr)
        
        expression_str = str(final_sympy_expr)

        # 5. Generate predictions using the fitted model
        predictions = self.model.predict(X_transformed)

        # 6. Extract complexity details
        details = {}
        try:
            best_equation_details = self.model.get_best()
            complexity = best_equation_details.get('complexity')
            if complexity is not None:
                details["complexity"] = complexity
        except (IndexError, KeyError):
            pass

        # 7. Return the final result
        return {
            "expression": expression_str,
            "predictions": predictions.tolist(),
            "details": details
        }