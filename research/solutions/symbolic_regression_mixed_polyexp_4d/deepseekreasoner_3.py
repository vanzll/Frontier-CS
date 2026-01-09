import numpy as np
from pysr import PySRRegressor
import sympy

class Solution:
    def __init__(self, **kwargs):
        pass

    def solve(self, X: np.ndarray, y: np.ndarray) -> dict:
        # Configure PySR with parameters suitable for 4D problem on CPU environment
        model = PySRRegressor(
            niterations=150,
            binary_operators=["+", "-", "*", "/", "**"],
            unary_operators=["sin", "cos", "exp", "log"],
            populations=25,
            population_size=50,
            maxsize=35,
            verbosity=0,
            progress=False,
            random_state=42,
            multithreading=True,
            ncycles_per_iteration=700,
            constraints={
                "**": (9, 2),  # Limit exponentiation complexity
                "log": 5,      # Limit log arguments complexity
                "exp": 5       # Limit exp arguments complexity
            },
            warm_start=True,
            batching=True,
            batch_size=1000,
            deterministic=True,
            precision=32,
            early_stop_condition=(
                "stop_if(loss, complexity) = loss < 1e-10 && complexity < 20"
            ),
            extra_sympy_mappings={"log": sympy.log if hasattr(sympy, 'log') else lambda x: sympy.log(x)},
        )
        
        # Fit model
        model.fit(X, y, variable_names=["x1", "x2", "x3", "x4"])
        
        # Get best expression
        try:
            best_expr = model.sympy()
            # Convert to string and ensure proper formatting
            expression = str(best_expr).replace("**", "^").replace("^", "**")
            # Replace any np.* functions if present
            expression = expression.replace("np.", "")
        except:
            # Fallback: use best equation from equations dataframe
            equations = model.equations_
            best_eq = equations[equations["loss"].idxmin()]
            expression = best_eq["equation"]
            # Clean up expression
            expression = expression.replace("**", "^").replace("^", "**")
            expression = expression.replace("np.", "")
        
        # Generate predictions
        try:
            predictions = model.predict(X).tolist()
        except:
            # Fallback: evaluate expression manually
            x1, x2, x3, x4 = X[:, 0], X[:, 1], X[:, 2], X[:, 3]
            safe_dict = {
                "x1": x1, "x2": x2, "x3": x3, "x4": x4,
                "sin": np.sin, "cos": np.cos, "exp": np.exp, "log": np.log
            }
            try:
                predictions = eval(expression, {"__builtins__": None}, safe_dict).tolist()
            except:
                # Final fallback: use linear regression
                A = np.column_stack([x1, x2, x3, x4, np.ones_like(x1)])
                coeffs, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
                a, b, c, d, e = coeffs
                expression = f"{a:.10f}*x1 + {b:.10f}*x2 + {c:.10f}*x3 + {d:.10f}*x4 + {e:.10f}"
                predictions = (a * x1 + b * x2 + c * x3 + d * x4 + e).tolist()
        
        return {
            "expression": expression,
            "predictions": predictions,
            "details": {}
        }