import numpy as np
import sympy
from pysr import PySRRegressor

class Solution:
    def __init__(self, **kwargs):
        pass

    def solve(self, X: np.ndarray, y: np.ndarray) -> dict:
        x1, x2 = X[:, 0], X[:, 1]
        
        # Feature engineering: create radial coordinate features
        # This simplifies the search space for PySR, assuming radial symmetry.
        r_sq = x1**2 + x2**2
        r = np.sqrt(r_sq)
        
        X_engineered = np.column_stack([x1, x2, r, r_sq])

        # Operator complexities are set to guide PySR's search towards
        # the competition's scoring metric.
        bin_ops_complexity = {op: 2 for op in ["+", "-", "*", "/", "**"]}
        una_ops_complexity = {op: 1 for op in ["sin", "cos", "exp", "log"]}
        custom_complexity = {**bin_ops_complexity, **una_ops_complexity}

        model = PySRRegressor(
            niterations=200,          # High iteration count, relying on timeout
            populations=24,
            population_size=33,
            binary_operators=["+", "-", "*", "/", "**"],
            unary_operators=["sin", "cos", "exp", "log"],
            maxsize=30,               # Allow for reasonably complex expressions
            procs=8,                  # Utilize all available CPU cores
            random_state=42,
            verbosity=0,
            progress=False,
            complexity_of_operators=custom_complexity,
            timeout_in_seconds=580,   # Safety net to finish within time limits
            temp_equation_file=True,
        )
        
        try:
            model.fit(X_engineered, y, variable_names=["x1", "x2", "r", "r_sq"])
        except Exception:
            # Fallback to a simple constant if PySR encounters an error
            return {"expression": "0.0"}

        if not hasattr(model, 'equations_') or model.equations_.shape[0] == 0:
            # Fallback if PySR fails to find any equations
            expression = "0.0"
        else:
            sympy_expr_with_feats = model.sympy()

            if isinstance(sympy_expr_with_feats, (float, int)) or sympy_expr_with_feats.is_number:
                expression = str(float(sympy_expr_with_feats))
            else:
                s_x1, s_x2, s_r, s_r_sq = sympy.symbols("x1 x2 r r_sq")
                
                # Definitions of engineered features in terms of original variables
                r_definition = (s_x1**2 + s_x2**2)**0.5
                r_sq_definition = s_x1**2 + s_x2**2

                # Substitute engineered features back into the expression
                final_sympy_expr = sympy_expr_with_feats.subs([
                    (s_r_sq, r_sq_definition),
                    (s_r, r_definition)
                ])
                
                # Convert final sympy expression to a string
                expression = str(final_sympy_expr)

        return {
            "expression": expression,
        }