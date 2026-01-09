import numpy as np
from pysr import PySRRegressor

class Solution:
    def __init__(self, **kwargs):
        pass

    def solve(self, X: np.ndarray, y: np.ndarray) -> dict:
        """
        Finds a symbolic expression for the given data using PySR.
        """
        model = PySRRegressor(
            niterations=120,
            populations=24,
            population_size=50,
            procs=8,  # Use all available vCPUs
            maxsize=40,
            binary_operators=["+", "*", "-", "/"],
            unary_operators=["exp", "cos", "sin", "log"],
            nested_constraints={
                "exp": {"exp": 0, "log": 0},
                "log": {"exp": 0, "log": 0},
                "sin": {"sin": 0, "cos": 0},
                "cos": {"sin": 0, "cos": 0},
            },
            random_state=42,
            verbosity=0,
            progress=False,
            turbo=True,
            precision=32,
            model_selection="best",
        )

        try:
            model.fit(X, y, variable_names=["x1", "x2", "x3", "x4"])
        except Exception:
            # If PySR fails for any reason, proceed to the fallback.
            pass

        if not hasattr(model, 'equations_') or model.equations_.empty:
            # Fallback to linear regression if PySR fails or finds no equations.
            A = np.column_stack([X, np.ones(X.shape[0])])
            try:
                coeffs, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
                expression_parts = []
                var_names = ["x1", "x2", "x3", "x4"]
                for i in range(4):
                    if not np.isclose(coeffs[i], 0):
                        expression_parts.append(f"{coeffs[i]:.6f}*{var_names[i]}")
                if not np.isclose(coeffs[4], 0):
                    expression_parts.append(f"{coeffs[4]:.6f}")

                expression = " + ".join(expression_parts).replace("+ -", "- ")
                if not expression:
                    expression = "0.0"

                predictions = X @ coeffs[:4] + coeffs[4]
                details = {"fallback_reason": "PySR search failed or found no equations."}
            except np.linalg.LinAlgError:
                # Extremely unlikely fallback
                expression = f"{np.mean(y):.6f}"
                predictions = np.full_like(y, np.mean(y))
                details = {"fallback_reason": "Linear regression failed."}

            return {
                "expression": expression,
                "predictions": predictions.tolist(),
                "details": details,
            }

        # Successful PySR run
        best_sympy_expr = model.sympy()
        expression = str(best_sympy_expr)
        
        predictions = model.predict(X)

        best_idx = model.equations_.score.idxmax()
        complexity = model.equations_.iloc[best_idx].complexity

        return {
            "expression": expression,
            "predictions": predictions.tolist(),
            "details": {"complexity": int(complexity)},
        }