import numpy as np
from pysr import PySRRegressor
import sympy as sp

class Solution:
    def __init__(self, **kwargs):
        pass

    def solve(self, X: np.ndarray, y: np.ndarray) -> dict:
        model = PySRRegressor(
            niterations=40,
            binary_operators=["+", "-", "*", "/", "**"],
            unary_operators=["sin", "cos", "exp", "log"],
            populations=8,
            population_size=33,
            maxsize=30,
            verbosity=0,
            progress=False,
            random_state=42,
            temp_equation_file=True,
            tempdir="/tmp",
            complexity_of_constants=2,
            complexity_of_variables=2,
            warm_start=False,
            loss="L2DistLoss()",
            early_stop_condition=(
                "stop_if(loss, complexity) = (loss < 1e-9) && (complexity < 10)"
            ),
        )
        
        try:
            model.fit(X, y, variable_names=["x1", "x2"])
            
            if hasattr(model, 'sympy') and callable(model.sympy):
                sympy_expr = model.sympy()
                if sympy_expr is not None:
                    simplified = sp.simplify(sympy_expr)
                    expression = str(simplified)
                else:
                    expression = self._fallback_expression(X, y)
            else:
                expression = self._fallback_expression(X, y)
            
            predictions = model.predict(X) if hasattr(model, 'predict') else None
            
            complexity = 0
            if expression:
                complexity = self._compute_complexity(expression)
            
            return {
                "expression": expression,
                "predictions": predictions.tolist() if predictions is not None else None,
                "details": {"complexity": complexity}
            }
            
        except Exception as e:
            return self._fallback_solution(X, y)

    def _fallback_expression(self, X: np.ndarray, y: np.ndarray) -> str:
        x1, x2 = X[:, 0], X[:, 1]
        
        A = np.column_stack([
            x1, x2, x1**2, x2**2, x1*x2,
            np.sin(x1), np.cos(x1), np.sin(x2), np.cos(x2),
            np.sin(x1 + x2), np.cos(x1 + x2),
            np.ones_like(x1)
        ])
        
        try:
            coeffs, residuals, rank, s = np.linalg.lstsq(A, y, rcond=None)
            
            terms = []
            term_names = [
                "x1", "x2", "x1**2", "x2**2", "x1*x2",
                "sin(x1)", "cos(x1)", "sin(x2)", "cos(x2)",
                "sin(x1 + x2)", "cos(x1 + x2)", ""
            ]
            
            for i, (coeff, name) in enumerate(zip(coeffs, term_names)):
                if abs(coeff) > 1e-8:
                    if i == len(term_names) - 1:
                        terms.append(f"{coeff:.6f}")
                    else:
                        terms.append(f"{coeff:.6f}*{name}")
            
            if not terms:
                return "0"
            
            expression = " + ".join(terms).replace("+ -", "- ")
            return expression
            
        except:
            return "0.5*x1 + 0.5*x2"

    def _fallback_solution(self, X: np.ndarray, y: np.ndarray) -> dict:
        x1, x2 = X[:, 0], X[:, 1]
        
        A = np.column_stack([x1, x2, np.ones_like(x1)])
        coeffs, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
        a, b, c = coeffs
        
        expression = f"{a:.6f}*x1 + {b:.6f}*x2 + {c:.6f}"
        predictions = a * x1 + b * x2 + c
        
        return {
            "expression": expression,
            "predictions": predictions.tolist(),
            "details": {"complexity": 3}
        }

    def _compute_complexity(self, expression: str) -> int:
        try:
            expr = sp.sympify(expression)
            
            binary_ops = 0
            unary_ops = 0
            
            def count_ops(node):
                nonlocal binary_ops, unary_ops
                if isinstance(node, sp.Function):
                    unary_ops += 1
                elif isinstance(node, sp.Pow):
                    binary_ops += 1
                elif isinstance(node, (sp.Mul, sp.Add)):
                    binary_ops += max(0, len(node.args) - 1)
                
                for arg in node.args:
                    count_ops(arg)
            
            count_ops(expr)
            return 2 * binary_ops + unary_ops
        except:
            return 10