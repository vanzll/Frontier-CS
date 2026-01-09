import numpy as np

class Solution:
    def __init__(self, **kwargs):
        self.random_state = int(kwargs.get("random_state", 42))
        self.max_terms = int(kwargs.get("max_terms", 14))
        self.tol_rel_improve = float(kwargs.get("tol_rel_improve", 1e-4))
        self.min_terms_before_stop = int(kwargs.get("min_terms_before_stop", 6))

    @staticmethod
    def _make_poly_terms(x1, x2, x3, x4):
        n = x1.shape[0]
        one = np.ones(n, dtype=np.float64)

        xs = [x1, x2, x3, x4]
        x2s = [x * x for x in xs]
        x3s = [x2s[i] * xs[i] for i in range(4)]

        terms = []
        # constant
        terms.append((one, "1"))

        # linear
        for i, s in enumerate(["x1", "x2", "x3", "x4"]):
            terms.append((xs[i], s))

        # squares
        for i, s in enumerate(["x1", "x2", "x3", "x4"]):
            terms.append((x2s[i], f"{s}**2"))

        # cubes
        for i, s in enumerate(["x1", "x2", "x3", "x4"]):
            terms.append((x3s[i], f"{s}**3"))

        # pair products
        pair_idx = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
        pair_str = [("x1", "x2"), ("x1", "x3"), ("x1", "x4"), ("x2", "x3"), ("x2", "x4"), ("x3", "x4")]
        for (i, j), (si, sj) in zip(pair_idx, pair_str):
            terms.append((xs[i] * xs[j], f"{si}*{sj}"))

        # squared * other (xi^2 * xj, i != j)
        names = ["x1", "x2", "x3", "x4"]
        for i in range(4):
            for j in range(4):
                if i == j:
                    continue
                terms.append((x2s[i] * xs[j], f"{names[i]}**2*{names[j]}"))

        # triple products
        triple_idx = [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]
        triple_str = [("x1", "x2", "x3"), ("x1", "x2", "x4"), ("x1", "x3", "x4"), ("x2", "x3", "x4")]
        for (i, j, k), (si, sj, sk) in zip(triple_idx, triple_str):
            terms.append((xs[i] * xs[j] * xs[k], f"{si}*{sj}*{sk}"))

        # quad product
        terms.append((x1 * x2 * x3 * x4, "x1*x2*x3*x4"))

        return terms

    @staticmethod
    def _make_poly_terms_deg2(x1, x2, x3, x4):
        n = x1.shape[0]
        one = np.ones(n, dtype=np.float64)

        xs = [x1, x2, x3, x4]
        x2s = [x * x for x in xs]

        terms = []
        terms.append((one, "1"))
        for i, s in enumerate(["x1", "x2", "x3", "x4"]):
            terms.append((xs[i], s))
        for i, s in enumerate(["x1", "x2", "x3", "x4"]):
            terms.append((x2s[i], f"{s}**2"))

        pair_idx = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
        pair_str = [("x1", "x2"), ("x1", "x3"), ("x1", "x4"), ("x2", "x3"), ("x2", "x4"), ("x3", "x4")]
        for (i, j), (si, sj) in zip(pair_idx, pair_str):
            terms.append((xs[i] * xs[j], f"{si}*{sj}"))

        return terms

    @staticmethod
    def _make_exp_components(x1, x2, x3, x4):
        xs = [x1, x2, x3, x4]
        x2s = [x * x for x in xs]
        names = ["x1", "x2", "x3", "x4"]

        comps = []
        # all
        comps.append((x2s[0] + x2s[1] + x2s[2] + x2s[3], "x1**2 + x2**2 + x3**2 + x4**2"))

        # pairs of squares
        pair_idx = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
        for i, j in pair_idx:
            comps.append((x2s[i] + x2s[j], f"{names[i]}**2 + {names[j]}**2"))

        # singles
        for i in range(4):
            comps.append((x2s[i], f"{names[i]}**2"))

        return comps

    @staticmethod
    def _arg_str(scale, comp_str):
        if abs(scale - 1.0) < 1e-15:
            return f"-({comp_str})"
        return f"-{scale:.12g}*({comp_str})"

    @staticmethod
    def _format_coef(c):
        if not np.isfinite(c):
            return "0.0"
        return f"{float(c):.12g}"

    @staticmethod
    def _format_term(coef, poly_str):
        c = float(coef)
        if poly_str == "1":
            return Solution._format_coef(c)
        if abs(c - 1.0) < 5e-12:
            return f"({poly_str})"
        if abs(c + 1.0) < 5e-12:
            return f"-({poly_str})"
        return f"({Solution._format_coef(c)})*({poly_str})"

    @staticmethod
    def _sum_expr(parts):
        if not parts:
            return "0.0"
        expr = " + ".join(parts)
        return expr

    def _build_feature_matrix(self, X):
        x1 = X[:, 0].astype(np.float64, copy=False)
        x2 = X[:, 1].astype(np.float64, copy=False)
        x3 = X[:, 2].astype(np.float64, copy=False)
        x4 = X[:, 3].astype(np.float64, copy=False)

        poly_all = self._make_poly_terms(x1, x2, x3, x4)
        poly_deg2 = self._make_poly_terms_deg2(x1, x2, x3, x4)

        exp_comps = self._make_exp_components(x1, x2, x3, x4)
        scales = [0.5, 1.0, 2.0]

        n = X.shape[0]
        m = len(poly_all) + len(poly_deg2) * (len(exp_comps) * len(scales))
        Phi = np.empty((n, m), dtype=np.float64)
        metas = [None] * m  # (poly_str, exp_arg_str or None)

        col = 0
        for arr, s in poly_all:
            Phi[:, col] = arr
            metas[col] = (s, None)
            col += 1

        for comp_arr, comp_str in exp_comps:
            for scale in scales:
                arg_values = -float(scale) * comp_arr
                exp_vals = np.exp(arg_values)
                arg_str = self._arg_str(scale, comp_str)
                for p_arr, p_str in poly_deg2:
                    Phi[:, col] = p_arr * exp_vals
                    metas[col] = (p_str, arg_str)
                    col += 1

        return Phi, metas

    def _omp_fit(self, Phi, y):
        n = Phi.shape[0]
        rng = np.random.default_rng(self.random_state)

        if n <= 2:
            selected = [0]
            coef, _, _, _ = np.linalg.lstsq(Phi[:, selected], y, rcond=None)
            return selected, coef

        idx = rng.permutation(n)
        n_train = int(0.8 * n)
        n_train = min(max(n_train, 2), n - 1)
        tr = idx[:n_train]
        va = idx[n_train:]

        Xtr = Phi[tr]
        ytr = y[tr]
        Xva = Phi[va]
        yva = y[va]

        norms = np.linalg.norm(Xtr, axis=0)
        norms = np.maximum(norms, 1e-12)

        selected = [0]  # force intercept
        Xsel_tr = Xtr[:, selected]
        coef, _, _, _ = np.linalg.lstsq(Xsel_tr, ytr, rcond=None)
        resid = ytr - Xsel_tr @ coef

        best_selected = selected.copy()
        best_coef = coef.copy()
        best_mse = float(np.mean((yva - Xva[:, best_selected] @ best_coef) ** 2))

        prev_mse = best_mse
        no_improve_steps = 0

        max_terms = max(1, int(self.max_terms))
        for t in range(1, max_terms):
            corr = np.abs(Xtr.T @ resid) / norms
            corr[selected] = 0.0
            j = int(np.argmax(corr))
            if corr[j] <= 0.0 or j in selected:
                break

            selected.append(j)
            Xsel_tr = Xtr[:, selected]
            coef, _, _, _ = np.linalg.lstsq(Xsel_tr, ytr, rcond=None)
            resid = ytr - Xsel_tr @ coef

            val_pred = Xva[:, selected] @ coef
            mse = float(np.mean((yva - val_pred) ** 2))

            rel_improve = (prev_mse - mse) / (abs(prev_mse) + 1e-12)
            if mse < best_mse:
                best_mse = mse
                best_selected = selected.copy()
                best_coef = coef.copy()

            if t >= self.min_terms_before_stop:
                if rel_improve < self.tol_rel_improve:
                    no_improve_steps += 1
                else:
                    no_improve_steps = 0
                if no_improve_steps >= 2:
                    break

            prev_mse = mse

        # Refit on full data with best selection
        Xbest = Phi[:, best_selected]
        coef_full, _, _, _ = np.linalg.lstsq(Xbest, y, rcond=None)
        return best_selected, coef_full

    def _build_expression(self, selected, coef, metas):
        poly_parts = []
        exp_groups = {}  # arg_str -> list[(coef, poly_str)]
        for c, idx in zip(coef, selected):
            poly_str, arg_str = metas[idx]
            if not np.isfinite(c) or abs(float(c)) < 1e-12:
                continue
            if arg_str is None:
                poly_parts.append((float(c), poly_str))
            else:
                exp_groups.setdefault(arg_str, []).append((float(c), poly_str))

        parts = []

        if poly_parts:
            inner = [self._format_term(c, s) for (c, s) in poly_parts if abs(c) >= 1e-12]
            poly_expr = self._sum_expr(inner)
            if poly_expr != "0.0":
                parts.append(f"({poly_expr})")

        for arg_str in sorted(exp_groups.keys()):
            grp = exp_groups[arg_str]
            inner = [self._format_term(c, s) for (c, s) in grp if abs(c) >= 1e-12]
            inner_expr = self._sum_expr(inner)
            if inner_expr == "0.0":
                continue
            parts.append(f"(({inner_expr}))*exp({arg_str})")

        if not parts:
            return "0.0"
        return " + ".join(parts)

    def solve(self, X: np.ndarray, y: np.ndarray) -> dict:
        X = np.asarray(X)
        y = np.asarray(y).reshape(-1).astype(np.float64, copy=False)

        Phi, metas = self._build_feature_matrix(X)

        try:
            selected, coef = self._omp_fit(Phi, y)
        except Exception:
            # Fallback to least squares on a small set: constant + linear + squares + pairs (no exp)
            x1, x2, x3, x4 = X[:, 0], X[:, 1], X[:, 2], X[:, 3]
            A = np.column_stack([
                np.ones_like(x1),
                x1, x2, x3, x4,
                x1 * x1, x2 * x2, x3 * x3, x4 * x4,
                x1 * x2, x1 * x3, x1 * x4, x2 * x3, x2 * x4, x3 * x4
            ]).astype(np.float64, copy=False)
            coef_ls, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
            expr_terms = [
                (coef_ls[0], "1"),
                (coef_ls[1], "x1"), (coef_ls[2], "x2"), (coef_ls[3], "x3"), (coef_ls[4], "x4"),
                (coef_ls[5], "x1**2"), (coef_ls[6], "x2**2"), (coef_ls[7], "x3**2"), (coef_ls[8], "x4**2"),
                (coef_ls[9], "x1*x2"), (coef_ls[10], "x1*x3"), (coef_ls[11], "x1*x4"),
                (coef_ls[12], "x2*x3"), (coef_ls[13], "x2*x4"), (coef_ls[14], "x3*x4"),
            ]
            parts = [self._format_term(c, s) for (c, s) in expr_terms if np.isfinite(c) and abs(float(c)) >= 1e-12]
            expression = self._sum_expr(parts)
            predictions = (A @ coef_ls).astype(np.float64, copy=False)
            return {
                "expression": expression,
                "predictions": predictions.tolist(),
                "details": {"n_terms": int(np.sum(np.abs(coef_ls) >= 1e-12))}
            }

        expression = self._build_expression(selected, coef, metas)
        predictions = (Phi[:, selected] @ coef).astype(np.float64, copy=False)

        return {
            "expression": expression,
            "predictions": predictions.tolist(),
            "details": {"n_terms": int(len(selected))}
        }