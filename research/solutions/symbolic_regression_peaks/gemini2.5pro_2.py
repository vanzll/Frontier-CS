import numpy as np
from pysr import PySRRegressor

class Solution:
    def __init__(self, **kwargs):
        """
        This constructor is not used in this solution.
        The PySR model is initialized within the solve method.
        """
        pass

    def solve(self, X: np.ndarray, y: np.ndarray) -> dict:
        """
        Finds a symbolic expression for the given data using PySR.

        Args:
            X: Feature matrix of shape (n, 2).
            y: Target values of shape (n,).

        Returns:
            A dictionary with the symbolic expression and other details.
        """

        def _fallback_solution(X_fb: np.ndarray, y_fb: np.ndarray) -> dict:
            """
            Provides a linear regression model as a fallback if PySR fails.
            """
            x1, x2 = X_fb[:, 0], X_fb[:, 1]
            A = np.c_[x1, x2, np.ones_like(x1)]
            try:
                coeffs, _, _, _ = np.linalg.lstsq(A, y_fb, rcond=None)
                a, b, c = coeffs
            except np.linalg.LinAlgError:
                # Handle cases where least squares fails
                a, b, c = 0.0, 0.0, np.mean(y_fb) if y_fb.size > 0 else 0.0

            expression = f"{a:.8f}*x1 + {b:.8f}*x2 + {c:.8f}"
            predictions = a * x1 + b * x2 + c
            return {
                "expression": expression,
                "predictions": predictions.tolist(),
                "details": {},
            }

        try:
            # Configure PySRRegressor with parameters optimized for the problem and environment.
            # The "peaks" function is known to have exponential and polynomial terms,
            # so we allow for higher complexity and more search iterations.
            model = PySRRegressor(
                niterations=150,
                populations=24,           # Number of populations (good for 8 cores)
                population_size=50,       # Size of each population
                maxsize=40,               # Maximum complexity of expressions
                
                # Operators allowed in the expressions, including power for polynomials.
                binary_operators=["+", "-", "*", "/", "pow"],
                unary_operators=["sin", "cos", "exp", "log"],

                # Utilize all available CPU cores for parallel processing.
                procs=8,
                
                # For reproducibility
                random_state=42,

                # Suppress output during execution
                verbosity=0,
                progress=False,
                
                # Save progress to a file, which can help with long-running jobs.
                temp_equation_file=True,
                
                # The default loss (MSE) and model selection ('best') are suitable.
            )

            # Fit the model to the data, specifying variable names.
            model.fit(X, y, variable_names=["x1", "x2"])

            # Check if PySR found any valid equations.
            if len(model.equations_) == 0:
                return _fallback_solution(X, y)
            
            # Select the best equation based on score (accuracy vs. complexity).
            best_equation = model.get_best()
            
            # Convert the sympy expression to a string.
            expression_str = str(best_equation.sympy_format)

            # Generate predictions from the best model.
            predictions = model.predict(X)
            
            # Get the complexity of the final expression.
            complexity = best_equation.complexity

            return {
                "expression": expression_str,
                "predictions": predictions.tolist(),
                "details": {"complexity": complexity},
            }

        except Exception:
            # If any error occurs during the PySR process, use the fallback.
            return _fallback_solution(X, y)