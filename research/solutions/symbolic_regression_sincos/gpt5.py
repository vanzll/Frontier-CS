import numpy as np
from sympy import sympify, Add, Mul, Pow, sin as sym_sin, cos as sym_cos, exp as sym_exp, log as sym_log

class Solution:
    def __init__(self, **kwargs):
        self.max_terms = int(kwargs.get("max_terms", 4))
        self.min_rel_improve = float(kwargs.get("min_rel_improve", 1e-6))
        self.min_abs_improve = float(kwargs.get("min_abs_improve", 1e-10))
        self.random_state = int(kwargs.get("random_state", 42))

    def _format_number(self, x: float) -> str:
        # Format number with up to 12 significant digits, avoid "-0"
        if not np.isfinite(x):
            return "0"
        s = f"{x:.12g}"
        if s == "-0":
            s = "0"
        return s

    def _safe_eval_predictions(self, expression: str, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        env = {
            "__builtins__": {},
            "sin": np.sin,
            "cos": np.cos,
            "exp": np.exp,
            "log": np.log,
            "x1": x1,
            "x2": x2,
        }
        try:
            pred = eval(expression, env)
            if isinstance(pred, (int, float)):
                pred = np.full_like(x1, float(pred), dtype=float)
            return np.array(pred, dtype=float)
        except Exception:
            # Fallback to zeros if eval fails
            return np.zeros_like(x1, dtype=float)

    def _compute_complexity(self, expression: str) -> int:
        try:
            expr = sympify(expression, evaluate=False)
        except Exception:
            try:
                expr = sympify(expression)
            except Exception:
                return 0

        binary = 0
        unary = 0

        def visit(e):
            nonlocal binary, unary
            f = e.func
            if f == Add:
                # n-1 binary operations for Add with n args
                binary += max(len(e.args) - 1, 0)
            elif f == Mul:
                binary += max(len(e.args) - 1, 0)
            elif f == Pow:
                binary += 1
            elif f in (sym_sin, sym_cos, sym_exp, sym_log):
                unary += 1
            for arg in getattr(e, "args", []):
                visit(arg)

        visit(expr)
        return int(2 * binary + unary)

    def _build_features(self, x1: np.ndarray, x2: np.ndarray):
        features = []
        def add_feature(name, vec, base_complexity):
            vec = np.asarray(vec, dtype=float)
            # Skip near-constant features
            if not np.all(np.isfinite(vec)):
                return
            if np.var(vec) < 1e-14:
                return
            features.append((name, vec, base_complexity))

        # Variables
        add_feature("x1", x1, 0)
        add_feature("x2", x2, 0)

        # Unary trig of individual variables
        add_feature("sin(x1)", np.sin(x1), 1)
        add_feature("cos(x1)", np.cos(x1), 1)
        add_feature("sin(x2)", np.sin(x2), 1)
        add_feature("cos(x2)", np.cos(x2), 1)

        # Sums and differences inside trig
        add_feature("sin(x1 + x2)", np.sin(x1 + x2), 3)  # 1 unary + 1 binary inside => complexity 2*1 + 1 = 3
        add_feature("cos(x1 + x2)", np.cos(x1 + x2), 3)
        add_feature("sin(x1 - x2)", np.sin(x1 - x2), 3)
        add_feature("cos(x1 - x2)", np.cos(x1 - x2), 3)

        # Products of trig
        add_feature("sin(x1)*cos(x2)", np.sin(x1) * np.cos(x2), 4)  # 1 mul + 2 unaries => 2*1 + 2 = 4
        add_feature("cos(x1)*sin(x2)", np.cos(x1) * np.sin(x2), 4)
        add_feature("sin(x1)*sin(x2)", np.sin(x1) * np.sin(x2), 4)
        add_feature("cos(x1)*cos(x2)", np.cos(x1) * np.cos(x2), 4)

        # Basic polynomial interactions (optional)
        add_feature("x1*x2", x1 * x2, 2)

        return features

    def _lstsq_fit(self, A: np.ndarray, y: np.ndarray):
        coefs, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
        yhat = A @ coefs
        mse = float(np.mean((y - yhat) ** 2))
        return coefs, yhat, mse

    def _forward_selection(self, features, y: np.ndarray, max_terms: int):
        n = len(y)
        ones = np.ones(n, dtype=float)
        selected = []
        # Initial intercept-only model
        A = ones.reshape(-1, 1)
        coefs, yhat, best_mse = self._lstsq_fit(A, y)
        best_model = {"indices": [], "coefs": coefs, "mse": best_mse}

        for _ in range(max_terms):
            improved = False
            best_candidate = None
            best_candidate_coefs = None
            best_candidate_mse = None
            # Search over remaining features
            for j in range(len(features)):
                if j in selected:
                    continue
                Aj = np.column_stack([ones] + [features[idx][1] for idx in selected] + [features[j][1]])
                coefs_j, yhat_j, mse_j = self._lstsq_fit(Aj, y)
                # Check improvement
                improvement = best_mse - mse_j
                threshold = max(self.min_abs_improve, self.min_rel_improve * max(best_mse, 1e-12))
                if (best_candidate_mse is None) or (mse_j < best_candidate_mse - 1e-18):
                    best_candidate = j
                    best_candidate_coefs = coefs_j
                    best_candidate_mse = mse_j
                elif abs(mse_j - best_candidate_mse) <= 1e-12:
                    # Tie-breaker by feature base complexity
                    _, _, bc_new = features[j]
                    _, _, bc_old = features[best_candidate]
                    if bc_new < bc_old:
                        best_candidate = j
                        best_candidate_coefs = coefs_j
                        best_candidate_mse = mse_j

            if best_candidate is None:
                break

            improvement = best_mse - best_candidate_mse
            threshold = max(self.min_abs_improve, self.min_rel_improve * max(best_mse, 1e-12))
            if improvement > threshold:
                selected.append(best_candidate)
                best_mse = best_candidate_mse
                best_model = {
                    "indices": selected.copy(),
                    "coefs": best_candidate_coefs.copy(),
                    "mse": best_candidate_mse,
                }
                improved = True
            if not improved:
                break

        return best_model

    def _build_expression_from_model(self, features, model):
        indices = model["indices"]
        coefs = model["coefs"]
        # coefs[0] is intercept
        terms = []
        # Constant term
        c0 = coefs[0]
        if abs(c0) > 1e-12:
            terms.append(self._format_number(c0))

        # Other terms
        for k, j in enumerate(indices):
            coef = float(coefs[k + 1])
            if abs(coef) < 1e-12:
                continue
            name, _, _ = features[j]
            # If coefficient approximately 1 or -1, omit magnitude
            if abs(coef - 1.0) <= 1e-6:
                term_str = f"{name}"
                if len(terms) == 0:
                    terms.append(term_str)
                else:
                    terms.append(f"+ {term_str}")
            elif abs(coef + 1.0) <= 1e-6:
                term_str = f"{name}"
                if len(terms) == 0:
                    terms.append(f"- {term_str}")
                else:
                    terms.append(f"- {term_str}")
            else:
                coef_str = self._format_number(abs(coef))
                if len(terms) == 0:
                    sign = "-" if coef < 0 else ""
                    terms.append(f"{sign}{coef_str}*({name})")
                else:
                    sign = "-" if coef < 0 else "+"
                    terms.append(f"{sign} {coef_str}*({name})")

        if len(terms) == 0:
            expression = "0"
        else:
            # Join terms; ensure no duplicate signs
            expression = " ".join(terms)
            # Fix patterns like "+ -" or starting with "+ "
            expression = expression.strip()
            if expression.startswith("+ "):
                expression = expression[2:]
        return expression

    def _evaluate_candidates(self, X: np.ndarray, y: np.ndarray):
        x1 = X[:, 0]
        x2 = X[:, 1]
        candidates = [
            "sin(x1) + cos(x2)",
            "sin(x1) + sin(x2)",
            "cos(x1) + cos(x2)",
            "sin(x1)*cos(x2)",
            "cos(x1)*sin(x2)",
            "sin(x1 - x2)",
            "sin(x1 + x2)",
            "cos(x1 - x2)",
            "cos(x1 + x2)",
        ]
        best_expr = None
        best_mse = None
        for expr in candidates:
            pred = self._safe_eval_predictions(expr, x1, x2)
            mse = float(np.mean((y - pred) ** 2))
            if (best_mse is None) or (mse < best_mse):
                best_mse = mse
                best_expr = expr
        return best_expr, best_mse

    def solve(self, X: np.ndarray, y: np.ndarray) -> dict:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        if X.ndim != 2 or X.shape[1] != 2:
            raise ValueError("X must be of shape (n, 2)")
        n = X.shape[0]
        if y.shape[0] != n:
            raise ValueError("y must have the same number of rows as X")

        x1 = X[:, 0]
        x2 = X[:, 1]

        # Build feature library
        features = self._build_features(x1, x2)

        # Forward selection to keep expression simple
        model = self._forward_selection(features, y, max_terms=self.max_terms)
        expr_from_model = self._build_expression_from_model(features, model)
        preds_model = self._safe_eval_predictions(expr_from_model, x1, x2)
        mse_model = float(np.mean((y - preds_model) ** 2))

        # Try simple candidate expressions as a fallback/alternative
        cand_expr, cand_mse = self._evaluate_candidates(X, y)

        # Choose between model-based and candidate-based by MSE and complexity
        expr_choice = expr_from_model
        mse_choice = mse_model
        comp_choice = self._compute_complexity(expr_choice)

        if cand_expr is not None:
            comp_cand = self._compute_complexity(cand_expr)
            # Prefer lower MSE; if similar within tolerance, prefer lower complexity
            if cand_mse + 1e-10 < mse_choice or (abs(cand_mse - mse_choice) <= 1e-10 and comp_cand < comp_choice):
                expr_choice = cand_expr
                mse_choice = cand_mse
                comp_choice = comp_cand

        # Final predictions and complexity
        predictions = self._safe_eval_predictions(expr_choice, x1, x2)
        complexity = self._compute_complexity(expr_choice)

        return {
            "expression": expr_choice,
            "predictions": predictions.tolist(),
            "details": {"complexity": complexity},
        }