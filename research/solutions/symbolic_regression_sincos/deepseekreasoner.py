import numpy as np
from pysr import PySRRegressor

class Solution:
    def __init__(self, **kwargs):
        pass

    def solve(self, X: np.ndarray, y: np.ndarray) -> dict:
        model = PySRRegressor(
            niterations=30,
            binary_operators=["+", "-", "*", "/", "**"],
            unary_operators=["sin", "cos", "exp", "log"],
            populations=8,
            population_size=20,
            maxsize=20,
            verbosity=0,
            progress=False,
            random_state=42,
            model_selection="best",
            temp_equation_file=False,
            update=False
        )
        
        model.fit(X, y, variable_names=["x1", "x2"])
        
        best_expr = model.sympy()
        expression = str(best_expr)
        
        try:
            predictions = model.predict(X).tolist()
        except:
            x1, x2 = X[:, 0], X[:, 1]
            namespace = {"x1": x1, "x2": x2, "sin": np.sin, "cos": np.cos, 
                        "exp": np.exp, "log": np.log}
            predictions = eval(expression, namespace).tolist()
        
        complexity = len(str(best_expr).replace(" ", ""))
        
        return {
            "expression": expression,
            "predictions": predictions,
            "details": {"complexity": complexity}
        }