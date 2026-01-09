import numpy as np
import sympy
from pysr import PySRRegressor

class Solution:
    def __init__(self, **kwargs):
        pass

    def solve(self, X: np.ndarray, y: np.ndarray) -> dict:
        """
        Finds a symbolic expression for the ripple dataset.

        This solution uses a feature engineering approach. It observes that the
        target function appears to be radially symmetric. A new feature,
        r_squared = x1**2 + x2**2, is created. PySR is then used to find a
        1D symbolic expression for y as a function of r_squared. This
        significantly simplifies the search space for the symbolic regression
        algorithm, leading to a faster and more accurate discovery of the
        underlying equation. The final expression is constructed by substituting
        r_squared back with (x1**2 + x2**2).
        """
        # 1. Feature Engineering: Create a radial feature
        # This simplifies the problem from 2D to 1D, as the function
        # appears to be of the form f(x1^2 + x2^2).
        x1_data = X[:, 0]
        x2_data = X[:, 1]
        r_squared_data = (x1_data**2 + x2_data**2).reshape(-1, 1)

        # 2. Configure PySRRegressor for the simplified 1D problem
        # The parameters are tuned for a relatively simple search on a CPU-only environment.
        model = PySRRegressor(
            niterations=50,
            populations=20,
            population_size=50,
            # A focused set of operators is chosen based on the problem's nature.
            binary_operators=["+", "*", "-"],
            unary_operators=["sin", "cos"],
            maxsize=15,
            # Utilize all available virtual CPUs.
            procs=8,
            # Set a random state for reproducibility.
            random_state=42,
            # Suppress console output for a clean execution environment.
            verbosity=0,
            progress=False,
            # Use the default model selection which balances accuracy and complexity.
            model_selection="best",
        )

        # 3. Fit the model to find the relationship between r_squared and y.
        model.fit(r_squared_data, y, variable_names=["r2"])

        # 4. Handle the case where PySR fails to find any equations
        if not hasattr(model, 'equations_') or model.equations_.empty:
            # If no equation is found, fallback to a constant model (the mean of y).
            mean_y = np.mean(y)
            expression = f"{mean_y:.8f}"
            predictions = np.full_like(y, mean_y)
            details = {}
        else:
            # 5. Reconstruct the full expression in terms of x1 and x2
            # Get the best symbolic expression found for the 1D problem
            expr_r2_sympy = model.sympy()

            # Define sympy symbols to perform the substitution.
            x1_sym, x2_sym, r2_sym = sympy.symbols("x1 x2 r2")
            
            # Substitute 'r2' back with '(x1**2 + x2**2)' to get the final expression.
            final_expr_sympy = expr_r2_sympy.subs(r2_sym, (x1_sym**2 + x2_sym**2))
            
            # Convert the final sympy object into a Python-evaluable string.
            expression = str(final_expr_sympy)
            
            # 6. Generate predictions using the fitted model
            predictions = model.predict(r_squared_data)
            
            # 7. Extract optional details like complexity
            # The last equation in the DataFrame is the one selected by PySR.
            complexity = model.equations_.iloc[-1]['complexity']
            details = {"complexity": int(complexity)}

        return {
            "expression": expression,
            "predictions": predictions.tolist(),
            "details": details,
        }