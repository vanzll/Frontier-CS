import numpy as np
import math

class Solution:
    def __init__(self, **kwargs):
        self.max_terms = int(kwargs.get("max_terms", 6))
        self.bic_tol = float(kwargs.get("bic_tol", 1e-9))
        self.min_coef_tol = float(kwargs.get("min_coef_tol", 1e-10))
        self.random_state = kwargs.get("random_state", 42)
        np.random.seed(self.random_state)

    def _feature_args(self):
        return ["x1", "x2", "x1+x2", "x1-x2", "2*x1", "2*x2"]

    def _compute_arg_array(self, arg, x1, x2):
        if arg == "x1":
            return x1
        elif arg == "x2":
            return x2
        elif arg == "x1+x2":
            return x1 + x2
        elif arg == "x1-x2":
            return x1 - x2
        elif arg == "2*x1":
            return 2.0 * x1
        elif arg == "2*x2":
            return 2.0 * x2
        else:
            # Fallback, not expected
            return np.zeros_like(x1)

    def _build_features(self, X):
        x1 = X[:, 0]
        x2 = X[:, 1]
        feature_list = []
        Phi_cols = []
        for arg in self._feature_args():
            arr = self._compute_arg_array(arg, x1, x2)
            s = np.sin(arr)
            c = np.cos(arr)
            feature_list.append(("sin", arg))
            Phi_cols.append(s)
            feature_list.append(("cos", arg))
            Phi_cols.append(c)
        Phi = np.column_stack(Phi_cols) if Phi_cols else np.zeros((len(x1), 0))
        return Phi, feature_list

    def _ols_bic(self, A, y):
        # Solve least squares and compute BIC
        n = A.shape[0]
        k = A.shape[1]
        if k == 0:
            yhat = np.zeros_like(y)
            rss = float(np.sum((y - yhat) ** 2))
            sigma2 = max(rss / max(n, 1), 1e-30)
            bic = n * math.log(sigma2) + 0 * math.log(max(n, 2))
            return np.zeros((0,)), yhat, rss, bic
        beta, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
        yhat = A.dot(beta)
        rss = float(np.sum((y - yhat) ** 2))
        sigma2 = max(rss / max(n, 1), 1e-30)
        bic = n * math.log(sigma2) + k * math.log(max(n, 2))
        return beta, yhat, rss, bic

    def _forward_selection(self, Phi, y, max_terms, bic_tol):
        n = len(y)
        ones = np.ones(n)
        selected = []
        remaining = list(range(Phi.shape[1]))
        # Start with intercept only
        A0 = ones.reshape(-1, 1)
        beta0, yhat0, rss0, bic0 = self._ols_bic(A0, y)
        current_A = A0
        current_beta = beta0
        current_bic = bic0
        improved = True
        while improved and len(selected) < max_terms and remaining:
            best_bic = None
            best_idx = None
            best_beta = None
            # Try adding each remaining feature
            for idx in remaining:
                A_try = np.column_stack([Phi[:, selected_i] for selected_i in selected] + [Phi[:, idx], ones])
                beta_try, _, _, bic_try = self._ols_bic(A_try, y)
                if (best_bic is None) or (bic_try < best_bic - 1e-18) or (abs(bic_try - best_bic) <= 1e-18 and idx < (best_idx if best_idx is not None else idx + 1)):
                    best_bic = bic_try
                    best_idx = idx
                    best_beta = beta_try
            if best_bic is not None and best_bic < current_bic - bic_tol:
                selected.append(best_idx)
                remaining.remove(best_idx)
                current_bic = best_bic
                current_beta = best_beta
                # Backward elimination step
                improved_inner = True
                while improved_inner and len(selected) > 0:
                    improved_inner = False
                    best_bic_remove = current_bic
                    best_remove = None
                    best_beta_remove = None
                    for i, rm_idx in enumerate(selected):
                        reduced_sel = [s for j, s in enumerate(selected) if j != i]
                        A_try = np.column_stack([Phi[:, ii] for ii in reduced_sel] + [ones])
                        beta_try, _, _, bic_try = self._ols_bic(A_try, y)
                        if bic_try < best_bic_remove - bic_tol:
                            best_bic_remove = bic_try
                            best_remove = rm_idx
                            best_beta_remove = beta_try
                    if best_remove is not None:
                        selected.remove(best_remove)
                        remaining.append(best_remove)
                        current_bic = best_bic_remove
                        current_beta = best_beta_remove
                        improved_inner = True
            else:
                improved = False
        # Final re-fit
        A_final = np.column_stack([Phi[:, idx] for idx in selected] + [ones])
        beta_final, yhat_final, rss_final, bic_final = self._ols_bic(A_final, y)
        return selected, beta_final, yhat_final, rss_final, bic_final

    def _format_number(self, val):
        return f"{val:.12g}"

    def _near(self, a, b, rel=1e-6, abs_tol=1e-10):
        scale = max(abs(a), abs(b), 1.0)
        return abs(a - b) <= rel * scale + abs_tol

    def _simplify_and_build_expression(self, feature_list, selected_indices, coefs, intercept):
        # coefs corresponds to selected_indices; intercept is scalar
        # Build map for coefficients of sin/cos for each argument
        coef_map = {}
        args_set = set(self._feature_args() + ["x1+x2", "x1-x2"])
        for i, idx in enumerate(selected_indices):
            func, arg = feature_list[idx]
            key = (func, arg)
            coef_map[key] = coef_map.get(key, 0.0) + coefs[i]

        # Rewrite combinations of sin(x1+x2) & sin(x1-x2) to products
        products = []
        def pop_coef(func, arg):
            key = (func, arg)
            v = coef_map.get(key, 0.0)
            return v

        # sin pair to sin(x1)*cos(x2) or cos(x1)*sin(x2)
        a = pop_coef("sin", "x1+x2")
        b = pop_coef("sin", "x1-x2")
        maxab = max(abs(a), abs(b))
        if maxab > 1e-12:
            if self._near(a, b, rel=1e-6, abs_tol=1e-12):
                # A sin(s) + B sin(d) ~ (A+B) sin x1 cos x2
                C = a + b
                if abs(C) > self.min_coef_tol:
                    products.append(("sin(x1)*cos(x2)", C))
                coef_map[("sin", "x1+x2")] = 0.0
                coef_map[("sin", "x1-x2")] = 0.0
            elif self._near(a, -b, rel=1e-6, abs_tol=1e-12):
                # A sin(s) + B sin(d) ~ (A-B) cos x1 sin x2, with B ~ -A
                C = a - b
                if abs(C) > self.min_coef_tol:
                    products.append(("cos(x1)*sin(x2)", C))
                coef_map[("sin", "x1+x2")] = 0.0
                coef_map[("sin", "x1-x2")] = 0.0

        # cos pair to cos(x1)*cos(x2) or sin(x1)*sin(x2)
        a = pop_coef("cos", "x1+x2")
        b = pop_coef("cos", "x1-x2")
        maxab = max(abs(a), abs(b))
        if maxab > 1e-12:
            if self._near(a, b, rel=1e-6, abs_tol=1e-12):
                # a cos(s) + b cos(d) ~ (a+b) cos x1 cos x2
                C = a + b
                if abs(C) > self.min_coef_tol:
                    products.append(("cos(x1)*cos(x2)", C))
                coef_map[("cos", "x1+x2")] = 0.0
                coef_map[("cos", "x1-x2")] = 0.0
            elif self._near(a, -b, rel=1e-6, abs_tol=1e-12):
                # a cos(s) + b cos(d) ~ (-a+b) sin x1 sin x2
                C = -a + b
                if abs(C) > self.min_coef_tol:
                    products.append(("sin(x1)*sin(x2)", C))
                coef_map[("cos", "x1+x2")] = 0.0
                coef_map[("cos", "x1-x2")] = 0.0

        # Combine sin/cos pairs for each arg to use phase-shifted single sinusoid when beneficial
        terms = []  # list of (coeff, func, arg, phase | None)
        for arg in self._feature_args():
            a = coef_map.get(("sin", arg), 0.0)
            b = coef_map.get(("cos", arg), 0.0)
            # threshold small
            if abs(a) <= self.min_coef_tol and abs(b) <= self.min_coef_tol:
                continue
            if abs(b) <= self.min_coef_tol:
                # only sin
                terms.append((a, "sin", arg, None))
            elif abs(a) <= self.min_coef_tol:
                # only cos
                terms.append((b, "cos", arg, None))
            else:
                # Use R*sin(arg + phi) with a*sin(arg) + b*cos(arg)
                R = math.hypot(a, b)
                if R <= self.min_coef_tol:
                    continue
                phi = math.atan2(b, a)  # since sin(x+phi)=sin x cos phi + cos x sin phi
                # Simplify if phi approx 0 or approx pi/2 modulo 2*pi
                # Reduce phi to range [-pi, pi]
                twopi = 2.0 * math.pi
                phi_mod = ((phi + math.pi) % twopi) - math.pi
                abs_b_over_R = abs(b) / R
                abs_a_over_R = abs(a) / R
                if abs_b_over_R <= 1e-6:
                    # phi ~ 0 -> R*sin(arg)
                    terms.append((R, "sin", arg, None))
                elif abs_a_over_R <= 1e-6:
                    # phi ~ pi/2 -> R*cos(arg)
                    terms.append((R, "cos", arg, None))
                else:
                    # keep phase
                    terms.append((R, "sin", arg, phi_mod))

        # Build expression string
        term_strs = []

        # Product terms first
        for name, C in products:
            if abs(C) <= self.min_coef_tol:
                continue
            if self._near(C, 1.0):
                term = name
            elif self._near(C, -1.0):
                term = f"-{name}"
            else:
                term = f"{self._format_number(C)}*{name}"
            term_strs.append(term)

        # Phase/simple terms
        for coeff, func, arg, phase in terms:
            if abs(coeff) <= self.min_coef_tol:
                continue
            if phase is None:
                inner = f"{func}({arg})"
            else:
                if abs(phase) <= 1e-12:
                    inner = f"{func}({arg})"
                else:
                    if phase < 0:
                        inner = f"{func}({arg} - {self._format_number(abs(phase))})"
                    else:
                        inner = f"{func}({arg} + {self._format_number(phase)})"
            if self._near(coeff, 1.0):
                term = inner
            elif self._near(coeff, -1.0):
                term = f"-{inner}"
            else:
                term = f"{self._format_number(coeff)}*{inner}"
            term_strs.append(term)

        # Intercept term
        if abs(intercept) > self.min_coef_tol:
            term_strs.append(self._format_number(intercept))

        # Assemble with + and - correctly
        if not term_strs:
            expression = "0"
        else:
            expr = term_strs[0]
            out = expr
            for t in term_strs[1:]:
                if t.startswith("-"):
                    out += " - " + t[1:]
                else:
                    out += " + " + t
            expression = out

        return expression

    def _eval_expression(self, expression, X):
        x1 = X[:, 0]
        x2 = X[:, 1]
        env = {
            "__builtins__": {},
        }
        local_env = {
            "sin": np.sin,
            "cos": np.cos,
            "exp": np.exp,
            "log": np.log,
            "x1": x1,
            "x2": x2,
            "pi": math.pi,
            "E": math.e,
        }
        try:
            yhat = eval(expression, env, local_env)
        except Exception:
            # Fallback: return zeros if expression eval fails
            yhat = np.zeros_like(x1)
        return np.asarray(yhat, dtype=float)

    def _try_simple_candidates(self, X, y):
        # Evaluate a set of simple candidate expressions and select the best by MSE
        candidates = [
            "sin(x1)",
            "cos(x1)",
            "sin(x2)",
            "cos(x2)",
            "sin(x1) + sin(x2)",
            "sin(x1) + cos(x2)",
            "cos(x1) + sin(x2)",
            "cos(x1) + cos(x2)",
            "sin(x1)*cos(x2)",
            "cos(x1)*sin(x2)",
            "sin(x1)*sin(x2)",
            "cos(x1)*cos(x2)",
            "sin(x1 + x2)",
            "sin(x1 - x2)",
            "cos(x1 + x2)",
            "cos(x1 - x2)",
        ]
        best_expr = None
        best_mse = None
        best_pred = None
        for expr in candidates:
            yhat = self._eval_expression(expr, X)
            mse = float(np.mean((y - yhat) ** 2))
            if (best_mse is None) or (mse < best_mse):
                best_mse = mse
                best_expr = expr
                best_pred = yhat
        return best_expr, best_pred, best_mse

    def solve(self, X: np.ndarray, y: np.ndarray) -> dict:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n = X.shape[0]
        if X.ndim != 2 or X.shape[1] != 2 or y.ndim != 1 or y.shape[0] != n:
            # Fallback simple expression
            expression = "sin(x1) + cos(x2)"
            predictions = self._eval_expression(expression, X)
            return {
                "expression": expression,
                "predictions": predictions.tolist(),
                "details": {}
            }

        # Try simple candidates first
        simple_expr, simple_pred, simple_mse = self._try_simple_candidates(X, y)

        # Build features and perform forward selection with BIC
        Phi, feature_list = self._build_features(X)
        selected_indices, beta, yhat_final, rss_final, bic_final = self._forward_selection(
            Phi, y, max_terms=self.max_terms, bic_tol=self.bic_tol
        )

        # beta corresponds to selected + intercept
        if len(beta) >= 1:
            coefs = beta[:-1] if len(beta) > 1 else np.array([], dtype=float)
            intercept = float(beta[-1])
        else:
            coefs = np.array([], dtype=float)
            intercept = 0.0

        # Build expression from selected features with simplifications
        learned_expr = self._simplify_and_build_expression(feature_list, selected_indices, coefs, intercept)
        learned_pred = self._eval_expression(learned_expr, X)
        learned_mse = float(np.mean((y - learned_pred) ** 2))

        # Choose between simple candidate and learned model
        # Prefer the one with lower MSE; if MSEs are very close, prefer shorter expression string
        if simple_expr is not None:
            if simple_mse < learned_mse - 1e-10:
                final_expr = simple_expr
                final_pred = simple_pred
            elif abs(simple_mse - learned_mse) <= 1e-10 and len(simple_expr) < len(learned_expr):
                final_expr = simple_expr
                final_pred = simple_pred
            else:
                final_expr = learned_expr
                final_pred = learned_pred
        else:
            final_expr = learned_expr
            final_pred = learned_pred

        return {
            "expression": final_expr,
            "predictions": final_pred.tolist(),
            "details": {}
        }