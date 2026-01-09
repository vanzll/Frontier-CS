import numpy as np
import pandas as pd
import sympy
from pysr import PySRRegressor

class Solution:
    def __init__(self, **kwargs):
        pass

    def solve(self, X: np.ndarray, y: np.ndarray) -> dict:
        """
        Solves the symbolic regression problem using PySR.
        """
        random_state = 42
        
        # Subsample data if necessary to speed up evolution
        n_samples = X.shape[0]
        max_fit_samples = 5000
        if n_samples > max_fit_samples:
            rng = np.random.RandomState(random_state)
            indices = rng.choice(n_samples, max_fit_samples, replace=False)
            X_fit = X[indices]
            y_fit = y[indices]
        else:
            X_fit = X
            y_fit = y

        # Configure PySRRegressor
        # Optimized for 4D Mixed PolyExp problem on 8 vCPUs
        model = PySRRegressor(
            niterations=60,
            binary_operators=["+", "-", "*", "/"],
            unary_operators=["sin", "cos", "exp", "log"],
            populations=20,
            population_size=40,
            maxsize=40,
            verbosity=0,
            progress=False,
            random_state=random_state,
            procs=8,
            multiprocessing=True,
            model_selection="best",
        )

        try:
            # Fit the model
            model.fit(X_fit, y_fit, variable_names=["x1", "x2", "x3", "x4"])
            
            # Extract the best expression as a string
            best_expr = model.sympy()
            expression = str(best_expr)
            
            # Compute predictions on the full dataset
            predictions = model.predict(X)
            
            # Sanity check predictions
            if np.any(np.isnan(predictions)) or np.any(np.isinf(predictions)):
                raise ValueError("Model predictions contain NaN or Inf")

        except Exception:
            # Fallback to Linear Regression if symbolic regression fails
            # Add bias term
            X_aug = np.column_stack([X, np.ones(n_samples)])
            coeffs, _, _, _ = np.linalg.lstsq(X_aug, y, rcond=None)
            
            # Construct linear expression string
            # coeffs has 5 elements: w1, w2, w3, w4, bias
            terms = [f"({coeffs[i]} * x{i+1})" for i in range(4)]
            terms.append(str(coeffs[4]))
            expression = " + ".join(terms)
            
            predictions = X_aug @ coeffs

        return {
            "expression": expression,
            "predictions": predictions.tolist(),
            "details": {}
        }