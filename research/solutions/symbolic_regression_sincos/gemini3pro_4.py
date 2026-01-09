import numpy as np
import os
import tempfile
import uuid
import sympy
from pysr import PySRRegressor

class Solution:
    def __init__(self, **kwargs):
        pass

    def solve(self, X: np.ndarray, y: np.ndarray) -> dict:
        """
        Fits a symbolic regression model to the data (X, y) to find a closed-form expression.
        Uses PySRRegressor with trigonometric functions enabled.
        """
        # Data subsampling for efficiency on large datasets
        # Limit fitting to 2000 samples to keep runtime within reasonable bounds
        n_samples = X.shape[0]
        max_fit_samples = 2000
        
        if n_samples > max_fit_samples:
            rng = np.random.RandomState(42)
            indices = rng.choice(n_samples, max_fit_samples, replace=False)
            X_fit = X[indices]
            y_fit = y[indices]
        else:
            X_fit = X
            y_fit = y

        # Create a unique temporary file path for the equation file
        # This prevents collisions when running multiple instances or in parallel
        temp_id = str(uuid.uuid4())
        equation_file = os.path.join(tempfile.gettempdir(), f"pysr_eq_{temp_id}.csv")

        # Configure PySRRegressor
        # - unary_operators includes sin/cos as per dataset hint
        # - niterations/populations tuned for 8 vCPUs environment
        model = PySRRegressor(
            niterations=100,
            binary_operators=["+", "-", "*", "/"],
            unary_operators=["sin", "cos", "exp", "log"],
            populations=20,
            population_size=33,
            maxsize=30,
            loss="loss(prediction, target) = (prediction - target)^2",
            verbosity=0,
            progress=False,
            deterministic=True,
            random_state=42,
            equation_file=equation_file
        )

        try:
            # Fit the model to the training data
            model.fit(X_fit, y_fit, variable_names=["x1", "x2"])

            # Retrieve the best expression (SymPy format) and convert to string
            best_expr = model.sympy()
            expression = str(best_expr)
            
            # Generate predictions for the full input dataset
            predictions = model.predict(X)

        except Exception:
            # Fallback: simple linear regression if symbolic regression fails
            # This ensures we always return a valid result
            x1, x2 = X[:, 0], X[:, 1]
            A = np.column_stack([x1, x2, np.ones_like(x1)])
            coeffs, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
            a, b, c = coeffs
            
            expression = f"{a}*x1 + {b}*x2 + {c}"
            predictions = A @ coeffs

        finally:
            # Cleanup temporary files created by PySR
            if os.path.exists(equation_file):
                try:
                    os.remove(equation_file)
                except OSError:
                    pass
            
            bkup_file = equation_file + ".bkup"
            if os.path.exists(bkup_file):
                try:
                    os.remove(bkup_file)
                except OSError:
                    pass

        return {
            "expression": expression,
            "predictions": predictions.tolist(),
            "details": {}
        }