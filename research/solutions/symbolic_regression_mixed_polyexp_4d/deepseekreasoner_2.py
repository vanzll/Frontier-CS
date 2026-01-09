import numpy as np
from pysr import PySRRegressor

class Solution:
    def __init__(self, **kwargs):
        pass

    def solve(self, X: np.ndarray, y: np.ndarray) -> dict:
        model = PySRRegressor(
            niterations=80,
            binary_operators=["+", "-", "*", "/", "**"],
            unary_operators=["sin", "cos", "exp", "log"],
            populations=15,
            population_size=35,
            maxsize=35,
            verbosity=0,
            progress=False,
            random_state=42,
            deterministic=True,
            early_stop_condition=1e-8,
            timeout_in_seconds=300,
            max_evals=1000000,
            loss="L2DistLoss()",
            temp_equation_file=True,
            tempdir=".",
            update=False,
            ncycles_per_iteration=700,
            fraction_replaced=0.15,
            fraction_replaced_hof=0.03,
            should_optimize_constants=True,
            weight_optimize=0.01,
            weight_simplify=0.002,
            warm_start=True,
            batching=True,
            batch_size=min(500, len(y)),
        )
        
        try:
            model.fit(X, y, variable_names=["x1", "x2", "x3", "x4"])
            expression = str(model.sympy())
            predictions = model.predict(X).tolist()
        except Exception:
            x1, x2, x3, x4 = X[:, 0], X[:, 1], X[:, 2], X[:, 3]
            A = np.column_stack([x1, x2, x3, x4, x1*x2, x1*x3, x1*x4, x2*x3, x2*x4, x3*x4,
                                 x1**2, x2**2, x3**2, x4**2, np.exp(-x1**2), np.exp(-x2**2),
                                 np.exp(-x3**2), np.exp(-x4**2), np.ones_like(x1)])
            coeffs, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
            terms = [
                f"{coeffs[0]:.6f}*x1", f"{coeffs[1]:.6f}*x2", f"{coeffs[2]:.6f}*x3", f"{coeffs[3]:.6f}*x4",
                f"{coeffs[4]:.6f}*x1*x2", f"{coeffs[5]:.6f}*x1*x3", f"{coeffs[6]:.6f}*x1*x4",
                f"{coeffs[7]:.6f}*x2*x3", f"{coeffs[8]:.6f}*x2*x4", f"{coeffs[9]:.6f}*x3*x4",
                f"{coeffs[10]:.6f}*x1**2", f"{coeffs[11]:.6f}*x2**2", f"{coeffs[12]:.6f}*x3**2",
                f"{coeffs[13]:.6f}*x4**2", f"{coeffs[14]:.6f}*exp(-x1**2)", f"{coeffs[15]:.6f}*exp(-x2**2)",
                f"{coeffs[16]:.6f}*exp(-x3**2)", f"{coeffs[17]:.6f}*exp(-x4**2)", f"{coeffs[18]:.6f}"
            ]
            expression = " + ".join(terms)
            predictions = A @ coeffs
        
        return {
            "expression": expression,
            "predictions": predictions,
            "details": {}
        }