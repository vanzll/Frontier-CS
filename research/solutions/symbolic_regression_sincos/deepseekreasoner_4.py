import numpy as np
from pysr import PySRRegressor

class Solution:
    def __init__(self, **kwargs):
        pass

    def solve(self, X: np.ndarray, y: np.ndarray) -> dict:
        model = PySRRegressor(
            niterations=50,
            binary_operators=["+", "-", "*", "/"],
            unary_operators=["sin", "cos", "exp", "log"],
            populations=8,
            population_size=30,
            maxsize=20,
            verbosity=0,
            progress=False,
            random_state=42,
            ncycles_per_iteration=500,
            early_stop_condition=1e-8,
            maxdepth=10,
            model_selection="best",
        )
        
        model.fit(X, y, variable_names=["x1", "x2"])
        
        best_expr = model.sympy()
        if best_expr is None:
            best_expr = model.equations_.iloc[-1]["sympy_format"]
        expression = str(best_expr).replace("**", "^").replace("^", "**")
        
        predictions = model.predict(X)
        
        complexity = 0
        expr_str = expression
        for op in ["+", "-", "*", "/"]:
            complexity += expr_str.count(op)
        complexity += expr_str.count("**")
        for func in ["sin", "cos", "exp", "log"]:
            complexity += expr_str.count(func)
            
        return {
            "expression": expression,
            "predictions": predictions.tolist(),
            "details": {"complexity": complexity}
        }