import numpy as np
import sympy as sp

class Solution:
    def __init__(self, **kwargs):
        self.max_terms = kwargs.get("max_terms", 3)
        self.topk_single = kwargs.get("topk_single", 10)
        self.topk_pairs = kwargs.get("topk_pairs", 15)
        self.random_state = kwargs.get("random_state", 42)
        self.rounding_rel_tol = kwargs.get("rounding_rel_tol", 0.01)
        self.rounding_abs_tol = kwargs.get("rounding_abs_tol", 0.01)
        self.mse_increase_ratio_tol = kwargs.get("mse_increase_ratio_tol", 0.005)
        self.mse_increase_abs_tol = kwargs.get("mse_increase_abs_tol", 1e-10)

    def _build_feature_bank(self, x1, x2):
        features = []
        def add(name, arr):
            if np.any(np.isfinite(arr)):
                features.append((name, arr))
        add("sin(x1)", np.sin(x1))
        add("cos(x1)", np.cos(x1))
        add("sin(x2)", np.sin(x2))
        add("cos(x2)", np.cos(x2))
        add("sin(x1 + x2)", np.sin(x1 + x2))
        add("sin(x1 - x2)", np.sin(x1 - x2))
        add("cos(x1 + x2)", np.cos(x1 + x2))
        add("cos(x1 - x2)", np.cos(x1 - x2))
        sx1 = np.sin(x1); cx1 = np.cos(x1); sx2 = np.sin(x2); cx2 = np.cos(x2)
        add("sin(x1)*sin(x2)", sx1 * sx2)
        add("sin(x1)*cos(x2)", sx1 * cx2)
        add("cos(x1)*sin(x2)", cx1 * sx2)
        add("cos(x1)*cos(x2)", cx1 * cx2)
        add("sin(2*x1)", np.sin(2 * x1))
        add("cos(2*x1)", np.cos(2 * x1))
        add("sin(2*x2)", np.sin(2 * x2))
        add("cos(2*x2)", np.cos(2 * x2))
        add("x1", x1)
        add("x2", x2)
        return features

    def _fit_ls(self, A, y, include_intercept=True):
        if include_intercept:
            A_ext = np.column_stack([A, np.ones(A.shape[0])])
        else:
            A_ext = A
        coef, _, _, _ = np.linalg.lstsq(A_ext, y, rcond=None)
        if include_intercept:
            w = coef[:-1]
            b = coef[-1]
        else:
            w = coef
            b = 0.0
        y_pred = A @ w + b
        mse = float(np.mean((y - y_pred) ** 2))
        return w, b, y_pred, mse

    def _evaluate_simple_expressions(self, X, y):
        # Evaluate a set of very simple expressions without fitted coefficients.
        x1 = X[:, 0]
        x2 = X[:, 1]
        simple_exprs = [
            "sin(x1)", "cos(x1)", "sin(x2)", "cos(x2)",
            "sin(x1) + sin(x2)", "sin(x1) + cos(x2)",
            "cos(x1) + sin(x2)", "cos(x1) + cos(x2)",
            "sin(x1)*cos(x2)", "cos(x1)*sin(x2)",
            "sin(x1)*sin(x2)", "cos(x1)*cos(x2)",
            "sin(x1 + x2)", "sin(x1 - x2)",
            "cos(x1 + x2)", "cos(x1 - x2)"
        ]
        env = {"sin": np.sin, "cos": np.cos, "exp": np.exp, "log": np.log, "x1": x1, "x2": x2}
        results = []
        for expr in simple_exprs:
            try:
                pred = eval(expr, {"__builtins__": {}}, env)
                mse = float(np.mean((y - pred) ** 2))
                results.append((expr, pred, mse))
            except Exception:
                continue
        return results

    def _select_candidates(self, features, y):
        n = len(features)
        # Fit singles
        single_results = []
        for i in range(n):
            A = features[i][1].reshape(-1, 1)
            w, b, pred, mse = self._fit_ls(A, y, include_intercept=True)
            single_results.append({
                "idxs": [i],
                "coefs": w.copy(),
                "intercept": float(b),
                "mse": mse,
                "pred": pred
            })
        single_results.sort(key=lambda d: d["mse"])
        topk_single = single_results[:min(self.topk_single, len(single_results))]

        # Fit pairs from topk singles
        pair_results = []
        top_indices = [d["idxs"][0] for d in topk_single]
        for i in range(len(top_indices)):
            for j in range(i + 1, len(top_indices)):
                idxi = top_indices[i]
                idxj = top_indices[j]
                A = np.column_stack([features[idxi][1], features[idxj][1]])
                w, b, pred, mse = self._fit_ls(A, y, include_intercept=True)
                pair_results.append({
                    "idxs": [idxi, idxj],
                    "coefs": w.copy(),
                    "intercept": float(b),
                    "mse": mse,
                    "pred": pred
                })
        pair_results.sort(key=lambda d: d["mse"])
        pair_results = pair_results[:min(self.topk_pairs, len(pair_results))]

        # Fit triples from top-6 single features to limit combinations
        triple_results = []
        top6 = top_indices[:min(6, len(top_indices))]
        for a in range(len(top6)):
            for b in range(a + 1, len(top6)):
                for c in range(b + 1, len(top6)):
                    i, j, k = top6[a], top6[b], top6[c]
                    A = np.column_stack([features[i][1], features[j][1], features[k][1]])
                    w, b0, pred, mse = self._fit_ls(A, y, include_intercept=True)
                    triple_results.append({
                        "idxs": [i, j, k],
                        "coefs": w.copy(),
                        "intercept": float(b0),
                        "mse": mse,
                        "pred": pred
                    })
        triple_results.sort(key=lambda d: d["mse"])
        return single_results, pair_results, triple_results

    def _construct_expression(self, features, idxs, coefs, intercept):
        # Build a string expression out of selected features, coefficients, intercept.
        terms = []
        for coef, idx in zip(coefs, idxs):
            name = features[idx][0]
            c = float(coef)
            if np.isfinite(c) and abs(c) > 0:
                if abs(c - 1.0) < 1e-12:
                    terms.append(f"{name}")
                elif abs(c + 1.0) < 1e-12:
                    terms.append(f"-({name})")
                else:
                    terms.append(f"({c:.12g})*{name}")
        expr = ""
        if terms:
            expr = " + ".join(terms)
        if intercept is not None and np.isfinite(intercept) and abs(intercept) > 0:
            if expr:
                expr = f"{expr} + ({intercept:.12g})"
            else:
                expr = f"{intercept:.12g}"
        if expr.strip() == "":
            expr = "0"
        # Clean up '+ -(' patterns if any
        expr = expr.replace("+ -(", "- (")
        return expr

    def _estimate_complexity_from_model(self, features, idxs, coefs, intercept):
        # Estimate complexity: C = 2 * (#binary ops) + (#unary ops)
        unary_per_feature = {}
        binops_inside_feature = {}
        for name, _ in features:
            if name not in unary_per_feature:
                if name.startswith("sin(") or name.startswith("cos("):
                    unary = 1
                elif name.startswith("exp(") or name.startswith("log("):
                    unary = 1
                else:
                    unary = 0
                # internal binary ops: inside sin or cos, count +,-,* if present
                # Approximate based on name
                binops_int = 0
                if " + " in name or " - " in name:
                    # inside trig functions like sin(x1 + x2)
                    binops_int += 1
                if "2*x1" in name or "2*x2" in name:
                    binops_int += 1
                if ")*" in name or "*sin(" in name or "*cos(" in name:
                    # product of trig terms
                    binops_int += 1
                if "*" in name and ("x1" in name or "x2" in name) and (name.startswith("x1") or name.startswith("x2")):
                    binops_int += 1
                unary_per_feature[name] = unary
                binops_inside_feature[name] = binops_int

        # Count terms kept (non-zero coefficients)
        kept = []
        for coef, idx in zip(coefs, idxs):
            if abs(coef) > 0:
                kept.append(idx)
        if not kept and (intercept is None or abs(intercept) < 1e-12):
            return 0
        # unary count sum for kept features
        unary_ops = 0
        bin_ops = 0
        for idx in kept:
            name = features[idx][0]
            unary_ops += unary_per_feature.get(name, 0)
            bin_ops += binops_inside_feature.get(name, 0)
            # coefficient multiplication unless coefficient is exactly 1 or -1
            c = float(coefs[idxs.index(idx)])
            if not (abs(c - 1.0) < 1e-12 or abs(c + 1.0) < 1e-12):
                bin_ops += 1  # multiply by constant
        # additions between terms
        if len(kept) >= 2:
            bin_ops += (len(kept) - 1)
        # intercept addition
        if intercept is not None and abs(intercept) > 0:
            if len(kept) >= 1:
                bin_ops += 1
            else:
                # only intercept: no binary ops
                pass
        complexity = 2 * bin_ops + unary_ops
        return int(complexity)

    def _sympy_complexity(self, expr):
        try:
            x1, x2 = sp.symbols('x1 x2')
            local_dict = {'sin': sp.sin, 'cos': sp.cos, 'exp': sp.exp, 'log': sp.log, 'x1': x1, 'x2': x2}
            s = sp.sympify(expr, locals=local_dict)
        except Exception:
            return None
        unary_ops = 0
        binary_ops = 0

        def count_ops(e):
            nonlocal unary_ops, binary_ops
            if e.is_Atom:
                return
            f = e.func
            args = e.args
            if f == sp.Add:
                n = len(args)
                if n >= 2:
                    binary_ops += (n - 1)
            elif f == sp.Mul:
                n = len(args)
                if n >= 2:
                    binary_ops += (n - 1)
            elif f == sp.Pow:
                binary_ops += 1
            elif f in (sp.sin, sp.cos, sp.exp, sp.log):
                unary_ops += 1
            for a in args:
                count_ops(a)
        count_ops(s)
        return int(2 * binary_ops + unary_ops)

    def _try_rounding(self, features, model, y):
        # Try to simplify by snapping coefficients and intercept to simple values if MSE doesn't degrade too much.
        idxs = model["idxs"]
        coefs = model["coefs"].copy()
        intercept = float(model["intercept"])
        base_mse = float(model["mse"])
        A = np.column_stack([features[i][1] for i in idxs]) if idxs else np.zeros((len(y), 0))
        var_y = float(np.var(y)) if np.var(y) > 0 else 1.0

        def predict_mse(cfs, b):
            if cfs.size > 0:
                pred = A @ cfs + b
            else:
                pred = np.full_like(y, b, dtype=float)
            return float(np.mean((y - pred) ** 2))

        nice_vals = np.array([0.0, 0.25, 0.5, 1.0, 1.5, 2.0,
                              -0.25, -0.5, -1.0, -1.5, -2.0, 3.0, -3.0], dtype=float)

        # Remove tiny coefficients
        abs_tol = max(1e-8, 1e-6 * np.std(y) if np.std(y) > 0 else 1e-8)
        changed = False
        for i in range(len(coefs)):
            if abs(coefs[i]) < abs_tol:
                old = coefs[i]
                coefs[i] = 0.0
                mse = predict_mse(coefs, intercept)
                # Accept drop if MSE increase small
                if mse <= base_mse * (1 + self.mse_increase_ratio_tol) + self.mse_increase_abs_tol * var_y:
                    base_mse = mse
                    changed = True
                else:
                    coefs[i] = old

        # Try rounding each coefficient to a "nice" value
        for i in range(len(coefs)):
            c = coefs[i]
            # Skip zeros
            if abs(c) < 1e-18:
                continue
            # choose nearest nice value if close
            diffs = np.abs(nice_vals - c)
            j = int(np.argmin(diffs))
            cand = nice_vals[j]
            rel = (abs(c - cand) / max(1e-12, abs(c)))
            absd = abs(c - cand)
            if (rel <= self.rounding_rel_tol) or (absd <= self.rounding_abs_tol):
                old = c
                coefs[i] = float(cand)
                mse = predict_mse(coefs, intercept)
                if mse <= base_mse * (1 + self.mse_increase_ratio_tol) + self.mse_increase_abs_tol * var_y:
                    base_mse = mse
                    changed = True
                else:
                    coefs[i] = old

        # Try zeroing intercept if small
        if abs(intercept) < abs_tol:
            old_b = intercept
            intercept = 0.0
            mse = predict_mse(coefs, intercept)
            if mse <= base_mse * (1 + self.mse_increase_ratio_tol) + self.mse_increase_abs_tol * var_y:
                base_mse = mse
                changed = True
            else:
                intercept = old_b
        else:
            # try rounding intercept
            diffs = np.abs(nice_vals - intercept)
            j = int(np.argmin(diffs))
            cand = float(nice_vals[j])
            rel = (abs(intercept - cand) / max(1e-12, abs(intercept)))
            absd = abs(intercept - cand)
            if (rel <= self.rounding_rel_tol) or (absd <= self.rounding_abs_tol):
                old_b = intercept
                intercept = cand
                mse = predict_mse(coefs, intercept)
                if mse <= base_mse * (1 + self.mse_increase_ratio_tol) + self.mse_increase_abs_tol * var_y:
                    base_mse = mse
                    changed = True
                else:
                    intercept = old_b

        new_model = {
            "idxs": idxs.copy(),
            "coefs": coefs.copy(),
            "intercept": intercept,
            "mse": base_mse
        }
        return new_model, changed

    def _model_to_expression_and_metrics(self, features, model, X, y):
        expr = self._construct_expression(features, model["idxs"], model["coefs"], model["intercept"])
        complexity_est = self._sympy_complexity(expr)
        if complexity_est is None:
            complexity_est = self._estimate_complexity_from_model(features, model["idxs"], model["coefs"], model["intercept"])
        # Predictions via safe eval
        env = {"sin": np.sin, "cos": np.cos, "exp": np.exp, "log": np.log, "x1": X[:, 0], "x2": X[:, 1]}
        try:
            pred = eval(expr, {"__builtins__": {}}, env)
            mse = float(np.mean((y - pred) ** 2))
        except Exception:
            # Fallback to linear combo computation
            if model["idxs"]:
                A = np.column_stack([features[i][1] for i in model["idxs"]])
                pred = A @ model["coefs"] + model["intercept"]
            else:
                pred = np.full_like(y, model["intercept"], dtype=float)
            mse = float(np.mean((y - pred) ** 2))
        return expr, pred, mse, complexity_est

    def solve(self, X: np.ndarray, y: np.ndarray) -> dict:
        X = np.asarray(X)
        y = np.asarray(y).astype(float)
        x1 = X[:, 0]
        x2 = X[:, 1]

        # Feature bank
        features = self._build_feature_bank(x1, x2)

        # Candidate models by LS on combinations
        singles, pairs, triples = self._select_candidates(features, y)
        candidates = []
        if singles:
            candidates.extend(singles[:min(10, len(singles))])
        if pairs:
            candidates.extend(pairs[:min(10, len(pairs))])
        if triples:
            candidates.extend(triples[:min(10, len(triples))])

        # Best LS model by MSE
        best_model = None
        if candidates:
            best_model = min(candidates, key=lambda d: d["mse"])

        # Evaluate simple predefined expressions (no fitted coefficients)
        simple_results = self._evaluate_simple_expressions(X, y)
        simple_best = None
        if simple_results:
            simple_best = min(simple_results, key=lambda t: t[2])

        final_expr = None
        final_pred = None
        final_mse = float("inf")
        final_complexity = None

        # Consider simplified rounded model
        if best_model is not None:
            rounded_model, _ = self._try_rounding(features, best_model, y)
            expr1, pred1, mse1, comp1 = self._model_to_expression_and_metrics(features, rounded_model, X, y)
            expr2, pred2, mse2, comp2 = self._model_to_expression_and_metrics(features, best_model, X, y)
            # Choose between rounded and original by MSE primarily, then complexity
            if mse1 < mse2 - 1e-12:
                final_expr, final_pred, final_mse, final_complexity = expr1, pred1, mse1, comp1
            elif mse2 < mse1 - 1e-12:
                final_expr, final_pred, final_mse, final_complexity = expr2, pred2, mse2, comp2
            else:
                # Equal within tolerance: choose smaller complexity
                if (comp1 is not None and comp2 is not None and comp1 < comp2):
                    final_expr, final_pred, final_mse, final_complexity = expr1, pred1, mse1, comp1
                else:
                    final_expr, final_pred, final_mse, final_complexity = expr2, pred2, mse2, comp2

        # Compare with simple expression (no fitted coefficients) using tie-breaker on complexity
        if simple_best is not None:
            expr_simple, pred_simple, mse_simple = simple_best
            comp_simple = self._sympy_complexity(expr_simple)
            # choose by mse, if close within tolerance prefer lower complexity
            if final_expr is None:
                final_expr, final_pred, final_mse, final_complexity = expr_simple, pred_simple, mse_simple, comp_simple
            else:
                mse_tol = max(1e-12, 1e-4 * (np.var(y) if np.var(y) > 0 else 1.0))
                if mse_simple + mse_tol < final_mse:
                    final_expr, final_pred, final_mse, final_complexity = expr_simple, pred_simple, mse_simple, comp_simple
                elif abs(mse_simple - final_mse) <= mse_tol:
                    # prefer lower complexity
                    if (comp_simple is not None and final_complexity is not None and comp_simple < final_complexity):
                        final_expr, final_pred, final_mse, final_complexity = expr_simple, pred_simple, mse_simple, comp_simple

        # Fallback if nothing worked
        if final_expr is None:
            # Just mean of y
            b = float(np.mean(y))
            final_expr = f"{b:.12g}"
            final_pred = np.full_like(y, b, dtype=float)
            final_mse = float(np.mean((y - final_pred) ** 2))
            final_complexity = 0

        # Ensure predictions are list
        try:
            preds_list = final_pred.tolist()
        except Exception:
            # As a fallback, re-evaluate expression
            env = {"sin": np.sin, "cos": np.cos, "exp": np.exp, "log": np.log, "x1": X[:, 0], "x2": X[:, 1]}
            try:
                preds_list = eval(final_expr, {"__builtins__": {}}, env).tolist()
            except Exception:
                preds_list = np.full_like(y, np.mean(y), dtype=float).tolist()

        details = {"complexity": int(final_complexity) if final_complexity is not None else None, "mse": float(final_mse)}
        return {
            "expression": final_expr,
            "predictions": preds_list,
            "details": details
        }