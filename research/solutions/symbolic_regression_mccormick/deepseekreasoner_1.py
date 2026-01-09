import numpy as np
from pysr import PySRRegressor
import sympy as sp

class Solution:
    def __init__(self, **kwargs):
        pass

    def solve(self, X: np.ndarray, y: np.ndarray) -> dict:
        model = PySRRegressor(
            niterations=60,
            binary_operators=["+", "-", "*", "/", "**"],
            unary_operators=["sin", "cos", "exp", "log"],
            populations=8,
            population_size=40,
            maxsize=30,
            verbosity=0,
            progress=False,
            random_state=42,
            nested_constraints={"**": {"**": False}},
            deterministic=True,
            procs=8,
            multithreading=False,
            maxdepth=12,
            warm_start=True,
            weight_optimize=0.02,
            adaptive_parsimony_scaling=50.0,
            timeout_in_seconds=60 * 10,
            loss="L2DistLoss()",
            complexity_of_operators={"/": 2, "**": 3, "sin": 2, "cos": 2, "exp": 3, "log": 3},
        )
        
        model.fit(X, y, variable_names=["x1", "x2"])
        
        best_expr = model.sympy()
        if best_expr is not None:
            simplified = sp.simplify(best_expr)
            expression = str(simplified)
            expression = expression.replace("**", "**").replace("exp", "exp")
        else:
            expression = "0"
        
        predictions = model.predict(X)
        if predictions is None:
            x1, x2 = X[:, 0], X[:, 1]
            try:
                predictions = eval(expression, {"x1": x1, "x2": x2, "sin": np.sin, "cos": np.cos, 
                                               "exp": np.exp, "log": np.log})
            except:
                predictions = np.zeros_like(y)
        
        complexity = 0
        if best_expr is not None:
            expr_str = str(best_expr)
            complexity += expr_str.count('+') + expr_str.count('-') + expr_str.count('*') + expr_str.count('/')
            complexity += 2 * expr_str.count('**')
            complexity += expr_str.count('sin') + expr_str.count('cos') + expr_str.count('exp') + expr_str.count('log')
        
        return {
            "expression": expression,
            "predictions": predictions.tolist(),
            "details": {"complexity": complexity}
        }