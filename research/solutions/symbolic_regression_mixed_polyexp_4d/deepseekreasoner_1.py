import numpy as np
from pysr import PySRRegressor

class Solution:
    def __init__(self, **kwargs):
        pass

    def solve(self, X: np.ndarray, y: np.ndarray) -> dict:
        model = PySRRegressor(
            niterations=200,
            binary_operators=["+", "-", "*", "/", "**"],
            unary_operators=["sin", "cos", "exp", "log"],
            populations=25,
            population_size=50,
            maxsize=35,
            verbosity=0,
            progress=False,
            random_state=42,
            complexity_of_constants=0,
            maxdepth=None,
            turbo=True,
            deterministic=True,
            batching=False,
            batch_size=50,
            warm_start=True,
            weight_optimize=0.02,
            ncycles_per_iteration=1000,
            parsimony=0.01,
            use_frequency=True,
            use_frequency_in_tournament=True,
            adaptive_parsimony_scaling=100.0,
            weight_add_node=1.0,
            weight_insert_node=3.0,
            weight_delete_node=3.0,
            weight_do_nothing=1.0,
            weight_mutate_constant=0.0,
            weight_mutate_operator=1.0,
            weight_simplify=0.0,
            crossover_probability=0.066,
            topn=30,
            elementwise_loss="L2DistLoss()",
            loss_function=None,
            constraints={
                "**": (9, 1),
                "log": (1, 9),
                "exp": (9, 1),
                "/": (9, 1),
                "sin": (9, 9),
                "cos": (9, 9)
            }
        )
        
        model.fit(X, y, variable_names=["x1", "x2", "x3", "x4"])
        
        best_expr = model.sympy()
        expression = str(best_expr).replace("**", "^").replace("^", "**")
        predictions = model.predict(X)
        
        return {
            "expression": expression,
            "predictions": predictions.tolist(),
            "details": {}
        }