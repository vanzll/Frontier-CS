import numpy as np
from pysr import PySRRegressor

class Solution:
    def __init__(self, **kwargs):
        pass

    def solve(self, X: np.ndarray, y: np.ndarray) -> dict:
        model = PySRRegressor(
            niterations=40,
            binary_operators=["+", "-", "*", "/", "**"],
            unary_operators=["sin", "cos", "exp", "log"],
            populations=15,
            population_size=33,
            maxsize=25,
            verbosity=0,
            progress=False,
            random_state=42,
            ncycles_per_iteration=700,
            early_stop_condition="stop_if(loss, complexity) = (loss < 1e-9) && (complexity < 10)",
            timeout_in_seconds=60,
            deterministic=True,
            max_depth=10,
            weight_optimize=0.02,
            weight_simplify=0.02,
            update_test=False,
            batching=True,
            batch_size=100,
            warm_start=True,
            precision=64,
            constraints={
                "**": (9, 9),
                "log": 9,
                "exp": 9,
                "sin": 9,
                "cos": 9
            }
        )
        
        model.fit(X, y, variable_names=["x1", "x2"])
        
        best_expr = model.sympy()
        expression = str(best_expr).replace("**", "^").replace("^", "**")
        
        predictions = model.predict(X)
        
        binary_ops = expression.count('+') + expression.count('-') + expression.count('*') + expression.count('/') + expression.count('**')
        unary_ops = expression.count('sin') + expression.count('cos') + expression.count('exp') + expression.count('log')
        complexity = 2 * binary_ops + unary_ops
        
        return {
            "expression": expression,
            "predictions": predictions.tolist(),
            "details": {"complexity": complexity}
        }