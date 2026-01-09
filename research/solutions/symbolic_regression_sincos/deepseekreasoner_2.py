import numpy as np
from pysr import PySRRegressor

class Solution:
    def __init__(self, **kwargs):
        pass

    def solve(self, X: np.ndarray, y: np.ndarray) -> dict:
        model = PySRRegressor(
            niterations=100,
            binary_operators=["+", "-", "*", "/"],
            unary_operators=["sin", "cos", "exp", "log"],
            populations=8,
            population_size=50,
            maxsize=30,
            verbosity=0,
            progress=False,
            random_state=42,
            ncycles_per_iteration=700,
            early_stop_condition="stop_if(loss, complexity) = loss < 1e-9 && complexity < 10",
            temp_annihilation=1.0,
            temp_decay=0.99,
            use_frequency=True,
            use_frequency_in_tournament=True,
            adaptive_parsimony_scaling=200.0,
        )
        
        model.fit(X, y, variable_names=["x1", "x2"])
        
        best_expr = model.sympy()
        expression = str(best_expr).replace("**", "^").replace("^", "**")
        
        predictions = model.predict(X)
        
        return {
            "expression": expression,
            "predictions": predictions.tolist(),
            "details": {}
        }