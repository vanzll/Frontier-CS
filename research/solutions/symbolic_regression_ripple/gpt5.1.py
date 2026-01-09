import numpy as np

try:
    from pysr import PySRRegressor
    _HAS_PYSR = True
except Exception:
    _HAS_PYSR = False


class Solution:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def _evaluate_expression_on_X(self, expression: str, X: np.ndarray) -> np.ndarray:
        x1 = X[:, 0]
        x2 = X[:, 1]
        local_dict = {
            "x1": x1,
            "x2": x2,
            "sin": np.sin,
            "cos": np.cos,
            "exp": np.exp,
            "log": np.log,
            "pi": np.pi,
            "E": np.e,
        }
        try:
            preds = eval(expression, {"__builtins__": {}}, local_dict)
            preds = np.asarray(preds, dtype=float).reshape(-1)
            if preds.shape[0] == X.shape[0]:
                return preds
        except Exception:
            pass

        try:
            import sympy as sp

            x1_sym, x2_sym = sp.symbols("x1 x2")
            sym_expr = sp.sympify(expression)
            func = sp.lambdify(
                (x1_sym, x2_sym),
                sym_expr,
                modules={"sin": np.sin, "cos": np.cos, "exp": np.exp, "log": np.log},
            )
            preds = func(x1, x2)
            return np.asarray(preds, dtype=float).reshape(-1)
        except Exception:
            return np.full(X.shape[0], float(np.mean(x1) * 0.0))

    def _fallback_expression(self, X: np.ndarray, y: np.ndarray):
        x1 = X[:, 0]
        x2 = X[:, 1]
        y = np.asarray(y, dtype=float).reshape(-1)

        r2 = x1 * x1 + x2 * x2
        ones = np.ones_like(r2)
        r2_sq = r2 * r2

        k_candidates = [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 15.0, 20.0]

        best_mse = np.inf
        best_coefs = None
        best_k = None

        for k in k_candidates:
            s = np.sin(k * r2)
            c = np.cos(k * r2)
            A = np.column_stack([ones, r2, r2_sq, s, c, r2 * s, r2 * c])
            try:
                coefs, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
            except Exception:
                continue
            preds = A @ coefs
            mse = np.mean((y - preds) ** 2)
            if mse < best_mse:
                best_mse = mse
                best_coefs = coefs
                best_k = k

        if best_coefs is None:
            A = np.column_stack([x1, x2, np.ones_like(x1)])
            coefs, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
            a, b, c0 = coefs
            expression = f"{a:.12g}*x1 + {b:.12g}*x2 + {c0:.12g}"
            preds = A @ coefs
            return expression, preds

        c0, c1, c2, c3, c4, c5, c6 = best_coefs
        k = best_k

        terms = []

        def add_term(coef, expr):
            if abs(coef) < 1e-10:
                return
            coef_str = f"{coef:.12g}"
            if expr == "1":
                terms.append(coef_str)
            else:
                terms.append(f"({coef_str})*{expr}")

        r2_expr = "(x1*x1 + x2*x2)"
        add_term(c0, "1")
        add_term(c1, r2_expr)
        add_term(c2, f"{r2_expr}*{r2_expr}")
        add_term(c3, f"sin({k:.12g}*{r2_expr})")
        add_term(c4, f"cos({k:.12g}*{r2_expr})")
        add_term(c5, f"{r2_expr}*sin({k:.12g}*{r2_expr})")
        add_term(c6, f"{r2_expr}*cos({k:.12g}*{r2_expr})")

        if not terms:
            expression = "0.0"
        else:
            expression = " + ".join(terms)

        s = np.sin(k * r2)
        c = np.cos(k * r2)
        A_best = np.column_stack([ones, r2, r2_sq, s, c, r2 * s, r2 * c])
        preds = A_best @ best_coefs
        return expression, preds

    def solve(self, X: np.ndarray, y: np.ndarray) -> dict:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)

        n_samples, n_features = X.shape
        if n_features != 2:
            raise ValueError("X must have shape (n_samples, 2)")

        details = {}
        expression = None
        predictions = None

        use_pysr = _HAS_PYSR and self.kwargs.get("use_pysr", True)

        if use_pysr:
            try:
                n = n_samples
                if n <= 5000:
                    niterations = 80
                elif n <= 20000:
                    niterations = 60
                else:
                    niterations = 40

                model = PySRRegressor(
                    niterations=niterations,
                    binary_operators=["+", "-", "*", "/"],
                    unary_operators=["sin", "cos", "exp", "log"],
                    populations=20,
                    population_size=40,
                    maxsize=30,
                    verbosity=0,
                    progress=False,
                    random_state=42,
                    procs=0,
                )

                model.fit(X, y, variable_names=["x1", "x2"])

                best_expr = model.sympy()
                try:
                    from sympy import Expr  # type: ignore

                    if isinstance(best_expr, (list, tuple)):
                        best_expr = best_expr[0]
                except Exception:
                    if isinstance(best_expr, (list, tuple)):
                        best_expr = best_expr[0]

                expression = str(best_expr)

                try:
                    eq_df = getattr(model, "equations_", None)
                    if eq_df is not None and hasattr(eq_df, "columns") and "loss" in eq_df.columns:
                        best_row = eq_df.sort_values("loss", ascending=True).iloc[0]
                        if "complexity" in best_row:
                            details["complexity"] = int(best_row["complexity"])
                except Exception:
                    pass

                try:
                    preds = model.predict(X)
                    predictions = np.asarray(preds, dtype=float).reshape(-1)
                except Exception:
                    predictions = self._evaluate_expression_on_X(expression, X)

            except Exception:
                expression, predictions = self._fallback_expression(X, y)

        if expression is None or predictions is None:
            expression, predictions = self._fallback_expression(X, y)

        predictions = np.asarray(predictions, dtype=float).reshape(-1)
        if predictions.shape[0] != n_samples:
            expression, predictions = self._fallback_expression(X, y)
            predictions = np.asarray(predictions, dtype=float).reshape(-1)

        return {
            "expression": expression,
            "predictions": predictions.tolist(),
            "details": details,
        }