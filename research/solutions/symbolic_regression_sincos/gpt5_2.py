import numpy as np
import sympy as sp

class Solution:
    def __init__(self, **kwargs):
        self.max_pairs = int(kwargs.get("max_pairs", 6))
        self.rel_mse_tolerance = float(kwargs.get("rel_mse_tolerance", 0.02))
        self.phi_quant_tol = float(kwargs.get("phi_quant_tol", 1e-3))
        self.coef_round_tol = float(kwargs.get("coef_round_tol", 2e-2))
        self.effect_drop_ratio = float(kwargs.get("effect_drop_ratio", 1e-3))

    def solve(self, X: np.ndarray, y: np.ndarray) -> dict:
        x1 = X[:, 0].astype(float)
        x2 = X[:, 1].astype(float)
        n = len(y)
        if n == 0:
            expression = "0"
            return {"expression": expression, "predictions": [], "details": {"complexity": 0}}

        y = y.astype(float)
        y_std = float(np.std(y)) if np.std(y) > 0 else 1.0
        eps = 1e-12

        # Precompute base features
        s1 = np.sin(x1)
        c1 = np.cos(x1)
        s2 = np.sin(x2)
        c2 = np.cos(x2)
        ssum = np.sin(x1 + x2)
        sdiff = np.sin(x1 - x2)
        csum = np.cos(x1 + x2)
        cdiff = np.cos(x1 - x2)
        sc = s1 * c2
        cs = c1 * s2
        ss = s1 * s2
        cc = c1 * c2

        base_features = {
            "sin(x1)": s1,
            "cos(x1)": c1,
            "sin(x2)": s2,
            "cos(x2)": c2,
            "sin(x1+x2)": ssum,
            "sin(x1-x2)": sdiff,
            "cos(x1+x2)": csum,
            "cos(x1-x2)": cdiff,
            "sin(x1)*cos(x2)": sc,
            "cos(x1)*sin(x2)": cs,
            "sin(x1)*sin(x2)": ss,
            "cos(x1)*cos(x2)": cc,
            "x1": x1,
            "x2": x2,
        }

        feature_std = {k: float(np.std(v)) for k, v in base_features.items()}

        # Helper functions
        def num_to_str(v):
            # Avoid negative zero formatting
            if abs(v) < 1e-15:
                v = 0.0
            return f"{v:.12g}"

        def eval_expr_str(expr_str, x1, x2):
            env = {
                "x1": x1,
                "x2": x2,
                "sin": np.sin,
                "cos": np.cos,
                "exp": np.exp,
                "log": np.log,
            }
            return eval(expr_str, {"__builtins__": {}}, env)

        def compute_mse(pred):
            diff = y - pred
            return float(np.mean(diff * diff))

        def count_complexity(expr_str):
            try:
                expr = sp.sympify(expr_str)
            except Exception:
                # Fallback naive count
                unary = sum(expr_str.count(f) for f in ["sin(", "cos(", "exp(", "log("])
                # count '**' first
                stars = expr_str.count("**")
                # remove those from single '*' counting
                mult = expr_str.count("*") - 2 * stars
                plus = expr_str.count("+")
                minus = expr_str.count("-")
                div = expr_str.count("/")
                # minus counted includes unary minus, approximate subtract out numbers of '(-' patterns
                # but complexity is only optional; keep simple
                binary = stars + mult + plus + div
                return int(2 * binary + unary)

            allowed_funcs = {sp.sin, sp.cos, sp.exp, sp.log}

            def ops(e):
                binary = 0
                unary = 0
                if isinstance(e, sp.FunctionClass):
                    return 0, 0
                if isinstance(e, sp.Function):
                    if e.func in allowed_funcs:
                        unary += 1
                    for a in e.args:
                        b, u = ops(a)
                        binary += b
                        unary += u
                    return binary, unary
                if isinstance(e, sp.Add):
                    # n-1 binary operations for n args
                    args = e.args
                    for a in args:
                        b, u = ops(a)
                        binary += b
                        unary += u
                    binary += max(len(args) - 1, 0)
                    return binary, unary
                if isinstance(e, sp.Mul):
                    args = e.args
                    for a in args:
                        b, u = ops(a)
                        binary += b
                        unary += u
                    binary += max(len(args) - 1, 0)
                    return binary, unary
                if isinstance(e, sp.Pow):
                    b, u = ops(e.base)
                    binary += b
                    unary += u
                    b, u = ops(e.exp)
                    binary += b
                    unary += u
                    binary += 1
                    return binary, unary
                # other
                for a in getattr(e, "args", []):
                    b, u = ops(a)
                    binary += b
                    unary += u
                return binary, unary

            b, u = ops(expr)
            return int(2 * b + u)

        def build_sum_expression(terms_with_coeffs, intercept):
            # terms_with_coeffs: list of tuples (coef, term_str)
            # Remove tiny effect terms based on their std
            filtered = []
            for coef, term in terms_with_coeffs:
                std_t = feature_std.get(term, 1.0)
                if abs(coef) * std_t < self.effect_drop_ratio * y_std:
                    continue
                filtered.append((coef, term))
            # Drop intercept if small
            if abs(intercept) < max(1e-8, self.effect_drop_ratio * y_std):
                intercept_use = 0.0
            else:
                intercept_use = intercept

            if not filtered and intercept_use == 0.0:
                return "0"

            # Round coefficients near 1 or -1 for simplicity
            rounded = []
            for coef, term in filtered:
                if abs(coef - 1.0) <= self.coef_round_tol:
                    coef_use = 1.0
                elif abs(coef + 1.0) <= self.coef_round_tol:
                    coef_use = -1.0
                else:
                    coef_use = coef
                rounded.append((coef_use, term))

            # Build string
            expr_parts = []
            first = True
            for coef, term in rounded:
                sign = "+" if coef >= 0 else "-"
                abs_coef = abs(coef)
                if first:
                    # include sign only if negative
                    if coef < 0:
                        if abs(abs_coef - 1.0) <= 1e-12:
                            expr_parts.append(f"-{term}")
                        else:
                            expr_parts.append(f"-{num_to_str(abs_coef)}*{term}")
                    else:
                        if abs(abs_coef - 1.0) <= 1e-12:
                            expr_parts.append(f"{term}")
                        else:
                            expr_parts.append(f"{num_to_str(abs_coef)}*{term}")
                    first = False
                else:
                    if abs(abs_coef - 1.0) <= 1e-12:
                        expr_parts.append(f" {sign} {term}")
                    else:
                        expr_parts.append(f" {sign} {num_to_str(abs_coef)}*{term}")

            if intercept_use != 0.0:
                sign_c = "+" if intercept_use >= 0 else "-"
                val = abs(intercept_use)
                if first:
                    expr_parts.append(f"{num_to_str(intercept_use)}")
                else:
                    expr_parts.append(f" {sign_c} {num_to_str(val)}")

            return "".join(expr_parts)

        def combine_sin_cos_pair(a, b, var_name):
            # Returns tuple: list of (coef, term) and possibly using phase form if better
            # Try to express as R*sin(var + phi), then quantize phi to multiples of pi/2
            R = float(np.hypot(a, b))
            if R < 1e-15:
                return []

            phi = float(np.arctan2(b, a))  # a*sin(x)+b*cos(x)=R*sin(x+phi)
            # Quantize phi
            half_pi = np.pi / 2.0
            m = int(np.round(phi / half_pi))
            phi_q = m * half_pi
            m_mod = ((m % 4) + 4) % 4
            if abs(phi - phi_q) <= self.phi_quant_tol:
                # Simplify to sin/cos with possible sign
                if m_mod == 0:
                    # sin(x)
                    return [(R, f"sin({var_name})")]
                elif m_mod == 1:
                    # cos(x)
                    return [(R, f"cos({var_name})")]
                elif m_mod == 2:
                    # -sin(x)
                    return [(-R, f"sin({var_name})")]
                else:
                    # -cos(x)
                    return [(-R, f"cos({var_name})")]
            else:
                # Keep as R*sin(var + phi)
                # Build "sin(var +/- abs(phi))"
                if phi >= 0:
                    term = f"sin({var_name}+{num_to_str(phi)})"
                else:
                    term = f"sin({var_name}-{num_to_str(-phi)})"
                return [(R, term)]

        # Candidate list
        candidates = []
        added_exprs = set()

        def add_candidate(expr_str):
            expr_str_c = expr_str.replace(" ", "")
            if expr_str_c in added_exprs:
                return
            try:
                pred = eval_expr_str(expr_str, x1, x2)
                if not np.all(np.isfinite(pred)):
                    return
                mse = compute_mse(pred)
            except Exception:
                return
            comp = count_complexity(expr_str)
            candidates.append({"expression": expr_str, "mse": mse, "pred": pred, "complexity": comp})
            added_exprs.add(expr_str_c)

        # 1) Hard-coded simple forms
        simple_forms = [
            "sin(x1)+cos(x2)",
            "sin(x1)+sin(x2)",
            "cos(x1)+cos(x2)",
            "sin(x1)*cos(x2)",
            "sin(x1)*sin(x2)",
            "cos(x1)*cos(x2)",
            "cos(x1)*sin(x2)",
            "sin(x1+x2)",
            "sin(x1-x2)",
            "cos(x1+x2)",
            "cos(x1-x2)",
            "sin(x1)",
            "cos(x2)"
        ]
        for sf in simple_forms:
            add_candidate(sf)

        # 2) Single-term linear fit: y ~ a*phi + b
        ones = np.ones_like(y)
        for term, arr in base_features.items():
            A = np.column_stack([arr, ones])
            try:
                coef_ab, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
            except Exception:
                continue
            a, b0 = coef_ab
            # Build expression with simplification
            expr1 = build_sum_expression([(a, term)], b0)
            add_candidate(expr1)

        # 3) Two-term linear fit among top features
        # Rank features by absolute correlation
        corrs = []
        y_center = y - np.mean(y)
        ysd = np.std(y_center) + eps
        for term, arr in base_features.items():
            a_center = arr - np.mean(arr)
            denom = (np.std(a_center) + eps) * ysd
            if denom <= eps:
                r = 0.0
            else:
                r = float(abs(np.dot(a_center, y_center) / (len(y) * denom)))
            corrs.append((r, term))
        corrs.sort(reverse=True)
        top_terms = [t for _, t in corrs[:max(2, self.max_pairs))]]

        # Fit all pairs
        m = len(top_terms)
        for i in range(m):
            for j in range(i + 1, m):
                t1 = top_terms[i]
                t2 = top_terms[j]
                A = np.column_stack([base_features[t1], base_features[t2], ones])
                try:
                    coefs, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
                except Exception:
                    continue
                a1, a2, b0 = coefs

                # Standard expression
                expr_pair = build_sum_expression([(a1, t1), (a2, t2)], b0)
                add_candidate(expr_pair)

                # If t1,t2 are sin/cos of same variable, combine to phase
                def same_var_pair(u, v, var):
                    return (u == f"sin({var})" and v == f"cos({var})") or (u == f"cos({var})" and v == f"sin({var})")

                if same_var_pair(t1, t2, "x1"):
                    # Order to a*sin + b*cos
                    if t1 == "sin(x1)":
                        a = a1
                        b = a2 if t2 == "cos(x1)" else 0.0
                    else:
                        a = a2
                        b = a1
                    combined_terms = combine_sin_cos_pair(a, b, "x1")
                    expr_comb = build_sum_expression(combined_terms, b0)
                    add_candidate(expr_comb)

                if same_var_pair(t1, t2, "x2"):
                    if t1 == "sin(x2)":
                        a = a1
                        b = a2 if t2 == "cos(x2)" else 0.0
                    else:
                        a = a2
                        b = a1
                    combined_terms = combine_sin_cos_pair(a, b, "x2")
                    expr_comb = build_sum_expression(combined_terms, b0)
                    add_candidate(expr_comb)

                # Detect sin(x1+x2) pattern: a*SC + b*CS (+ c)
                if (t1 == "sin(x1)*cos(x2)" and t2 == "cos(x1)*sin(x2)") or (t2 == "sin(x1)*cos(x2)" and t1 == "cos(x1)*sin(x2)"):
                    # align coefficients
                    if t1 == "sin(x1)*cos(x2)":
                        a_sc = a1
                        a_cs = a2
                    else:
                        a_sc = a2
                        a_cs = a1
                    if abs(a_sc - a_cs) <= max(self.coef_round_tol, 0.02 * max(1.0, abs(a_sc), abs(a_cs))):
                        k = 0.5 * (a_sc + a_cs)
                        expr_sc = build_sum_expression([(k, "sin(x1+x2)")], b0)
                        add_candidate(expr_sc)

        # Choose best candidate by MSE, with preference for lower complexity within tolerance
        if not candidates:
            # Fallback to linear regression x1,x2
            A = np.column_stack([x1, x2, np.ones_like(x1)])
            coefs, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
            a, b, c = coefs
            expr = build_sum_expression([(a, "x1"), (b, "x2")], c)
            pred = eval_expr_str(expr, x1, x2)
            return {
                "expression": expr,
                "predictions": pred.tolist(),
                "details": {"complexity": count_complexity(expr)}
            }

        # Sort by MSE
        candidates.sort(key=lambda d: d["mse"])
        best_mse = candidates[0]["mse"]

        # Filter within tolerance
        tol = best_mse * (1.0 + self.rel_mse_tolerance) + 1e-18
        near_best = [c for c in candidates if c["mse"] <= tol]

        # Pick minimal complexity among near_best
        near_best.sort(key=lambda d: (d["complexity"], d["mse"]))
        best = near_best[0]

        return {
            "expression": best["expression"],
            "predictions": best["pred"].tolist(),
            "details": {"complexity": best["complexity"], "mse": best["mse"]}
        }