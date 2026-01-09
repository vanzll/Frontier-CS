import numpy as np
from pysr import PySRRegressor
import sympy

class Solution:
    def __init__(self, **kwargs):
        pass

    def solve(self, X: np.ndarray, y: np.ndarray) -> dict:
        model = PySRRegressor(
            niterations=100,
            binary_operators=["+", "-", "*", "/", "**"],
            unary_operators=["exp", "sin", "cos", "log"],
            populations=8,
            population_size=50,
            maxsize=30,
            parsimony=0.02,
            verbosity=0,
            progress=False,
            random_state=42,
            deterministic=True,
            early_stop_condition=(
                "stop_if(loss, complexity) = (loss < 1e-9) && (complexity < 30)"
            ),
            maxdepth=10,
            warm_start=False,
            temp_equation_file=True,
            temp_dir="/tmp/pysr_cache",
            ncyclesperiteration=1000,
        )
        
        try:
            model.fit(X, y, variable_names=["x1", "x2"])
            
            if model.equations_ is not None and len(model.equations_) > 0:
                best_eq = model.equations_.iloc[0]
                expression = str(model.sympy())
                
                try:
                    sympy_expr = sympy.sympify(expression)
                    predictions = model.predict(X)
                except:
                    predictions = None
                    
                complexity = int(best_eq['complexity'])
            else:
                expression = "x1 + x2"
                predictions = None
                complexity = 2
                
        except Exception:
            expression = "x1 + x2"
            predictions = None
            complexity = 2

        return {
            "expression": expression,
            "predictions": predictions.tolist() if predictions is not None else None,
            "details": {"complexity": complexity}
        }