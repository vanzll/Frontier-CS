import numpy as np
import itertools
import math

class Solution:
    def __init__(self, **kwargs):
        self.random_state = int(kwargs.get("random_state", 0))
        self.max_terms = int(kwargs.get("max_terms", 4))  # non-constant terms
        self.top_k = int(kwargs.get("top_k", 22))
        self.ridge = float(kwargs.get("ridge", 1e-10))

    @staticmethod
    def _snap_coef(c):
        if not np.isfinite(c):
            return c
        candidates = [0.0, 1.0, -1.0, 2.0, -2.0, 0.5, -0.5, 3.0, -3.0, 0.25, -0.25]
        ac = abs(c)
        tol = 1e-6 if ac < 5 else 1e-5
        for v in candidates:
            if abs(c - v) <= tol:
                return float(v)
        # snap to simple rationals if very close
        for den in (3, 4, 5, 6, 8, 10, 12):
            num = round(c * den)
            v = num / den
            if abs(c - v) <= 5e-7 * (1 + abs(v)):
                return float(v)
        return float(c)

    @staticmethod
    def _is_finite_col(col):
        return np.all(np.isfinite(col))

    @staticmethod
    def _col_var(col):
        m = np.mean(col)
        return float(np.mean((col - m) * (col - m)))

    @staticmethod
    def _safe_solve(G, b, ridge):
        k = G.shape[0]
        if k == 1:
            denom = G[0, 0] + ridge
            if denom == 0.0:
                return np.array([0.0], dtype=np.float64)
            return np.array([b[0] / denom], dtype=np.float64)
        A = G + ridge * np.eye(k, dtype=np.float64)
        try:
            return np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            return np.linalg.lstsq(A, b, rcond=None)[0]

    @staticmethod
    def _compute_corrs(Phi, y):
        y0 = y - np.mean(y)
        ystd = np.sqrt(np.mean(y0 * y0))
        if ystd == 0.0:
            return np.zeros(Phi.shape[1], dtype=np.float64)
        Phi0 = Phi - np.mean(Phi, axis=0, keepdims=True)
        Phi_std = np.sqrt(np.mean(Phi0 * Phi0, axis=0))
        denom = Phi_std * ystd
        denom[denom == 0.0] = np.inf
        corrs = np.abs((Phi0.T @ y0) / (Phi.shape[0] * denom))
        corrs[~np.isfinite(corrs)] = 0.0
        return corrs

    @staticmethod
    def _term_complexity(term_bin_unary):
        b, u = term_bin_unary
        return 2 * b + u

    def _build_library(self, X):
        x1 = X[:, 0].astype(np.float64, copy=False)
        x2 = X[:, 1].astype(np.float64, copy=False)

        terms = []
        cols = []
        comp = []  # (binary_ops, unary_ops)

        def add(name, col, bin_ops, unary_ops):
            col = np.asarray(col, dtype=np.float64)
            if col.ndim != 1 or col.shape[0] != X.shape[0]:
                return
            if not self._is_finite_col(col):
                return
            if self._col_var(col) < 1e-16:
                return
            terms.append(name)
            cols.append(col)
            comp.append((int(bin_ops), int(unary_ops)))

        # constant
        terms.append("1.0")
        cols.append(np.ones_like(x1, dtype=np.float64))
        comp.append((0, 0))

        # linear
        add("x1", x1, 0, 0)
        add("x2", x2, 0, 0)

        # trig basics
        s1 = np.sin(x1); c1 = np.cos(x1)
        s2 = np.sin(x2); c2 = np.cos(x2)
        add("sin(x1)", s1, 0, 1)
        add("cos(x1)", c1, 0, 1)
        add("sin(x2)", s2, 0, 1)
        add("cos(x2)", c2, 0, 1)

        for k in (2.0, 3.0):
            add(f"sin({k}*x1)", np.sin(k * x1), 1, 1)
            add(f"cos({k}*x1)", np.cos(k * x1), 1, 1)
            add(f"sin({k}*x2)", np.sin(k * x2), 1, 1)
            add(f"cos({k}*x2)", np.cos(k * x2), 1, 1)

        # combined arguments
        add("sin(x1+x2)", np.sin(x1 + x2), 1, 1)
        add("cos(x1+x2)", np.cos(x1 + x2), 1, 1)
        add("sin(x1-x2)", np.sin(x1 - x2), 1, 1)
        add("cos(x1-x2)", np.cos(x1 - x2), 1, 1)

        # products of trig
        add("sin(x1)*sin(x2)", s1 * s2, 1, 2)
        add("sin(x1)*cos(x2)", s1 * c2, 1, 2)
        add("cos(x1)*sin(x2)", c1 * s2, 1, 2)
        add("cos(x1)*cos(x2)", c1 * c2, 1, 2)

        Phi = np.column_stack(cols).astype(np.float64, copy=False)
        return terms, Phi, comp

    def _select_features(self, Phi_tr, y_tr, names, comp):
        m = Phi_tr.shape[1]
        corrs = self._compute_corrs(Phi_tr[:, 1:], y_tr)
        idx_sorted = np.argsort(-corrs) + 1

        mandatory = set()
        for i, nm in enumerate(names):
            if nm in ("x1", "x2", "sin(x1)", "cos(x1)", "sin(x2)", "cos(x2)"):
                mandatory.add(i)

        keep = [0]
        for i in idx_sorted[: max(0, self.top_k - 1)]:
            keep.append(int(i))
        for i in sorted(mandatory):
            if i not in keep:
                keep.append(i)

        keep = sorted(set(keep))
        Phi_red = Phi_tr[:, keep]
        names_red = [names[i] for i in keep]
        comp_red = [comp[i] for i in keep]
        return keep, Phi_red, names_red, comp_red

    def _score_subset(self, w, Gv_S, bv_S, yty_v, C_subset, n_terms_total, eps=1e-12):
        rss_v = float(yty_v - 2.0 * (w @ bv_S) + w @ (Gv_S @ w))
        if rss_v < 0.0 and rss_v > -1e-8:
            rss_v = 0.0
        mse_v = rss_v / max(1, int(self._n_val))
        # light complexity penalty
        crit = mse_v * (1.0 + 0.01 * max(0, C_subset - 8)) * (1.0 + 0.03 * max(0, n_terms_total - 3))
        return crit, mse_v, rss_v

    def _build_expression(self, names, coeffs):
        # names include constant as index 0 if present
        terms_out = []
        coeffs_out = []
        for nm, c in zip(names, coeffs):
            c = float(c)
            if not np.isfinite(c):
                continue
            c = self._snap_coef(c)
            if abs(c) < 1e-10:
                continue
            terms_out.append(nm)
            coeffs_out.append(c)

        if not terms_out:
            return "0.0", 0

        # Build string with minimal clutter.
        expr_parts = []
        for nm, c in zip(terms_out, coeffs_out):
            if nm == "1.0":
                expr_parts.append(f"{c:.12g}")
                continue
            if c == 1.0:
                expr_parts.append(f"{nm}")
            elif c == -1.0:
                expr_parts.append(f"-({nm})")
            else:
                expr_parts.append(f"({c:.12g})*({nm})")

        expr = expr_parts[0]
        for part in expr_parts[1:]:
            if part.startswith("-"):
                expr = f"({expr}) {part}"
            else:
                expr = f"({expr}) + ({part})"

        # Approx complexity: count ops from term comps + plus ops + coef multiplications
        # Here, compute from final coefficients/names.
        C = 0
        nonconst = 0
        for nm, c in zip(terms_out, coeffs_out):
            if nm != "1.0":
                nonconst += 1
                if c not in (1.0, -1.0):
                    C += 2  # multiplication by coefficient
        if len(terms_out) >= 2:
            C += 2 * (len(terms_out) - 1)  # additions
        return expr, C

    def solve(self, X: np.ndarray, y: np.ndarray) -> dict:
        X = np.asarray(X)
        y = np.asarray(y, dtype=np.float64).reshape(-1)
        n = X.shape[0]
        if n == 0:
            return {"expression": "0.0", "predictions": [], "details": {"complexity": 0}}

        names, Phi_full, comp_full = self._build_library(X)

        rng = np.random.RandomState(self.random_state)
        if n >= 50:
            perm = rng.permutation(n)
            n_val = max(1, int(0.2 * n))
            val_idx = perm[:n_val]
            tr_idx = perm[n_val:]
        else:
            tr_idx = np.arange(n)
            val_idx = np.arange(n)

        self._n_val = len(val_idx)

        Phi_tr = Phi_full[tr_idx]
        y_tr = y[tr_idx]
        Phi_val = Phi_full[val_idx]
        y_val = y[val_idx]

        keep, Phi_tr_red, names_red, comp_red = self._select_features(Phi_tr, y_tr, names, comp_full)
        Phi_val_red = Phi_val[:, [keep.index(i) for i in keep]] if False else Phi_val[:, keep]  # keep aligned

        # Precompute Gram matrices for fast subset evaluation
        Gt = Phi_tr_red.T @ Phi_tr_red
        bt = Phi_tr_red.T @ y_tr
        yty_t = float(y_tr @ y_tr)

        Gv = Phi_val_red.T @ Phi_val_red
        bv = Phi_val_red.T @ y_val
        yty_v = float(y_val @ y_val)

        m = Phi_tr_red.shape[1]

        best = {
            "crit": float("inf"),
            "mse_v": float("inf"),
            "S": (0,),
            "w": np.array([np.mean(y_tr)], dtype=np.float64),
            "C": 0,
            "k": 0,
        }

        # include constant always, but allow model without explicit constant if it helps (rare)
        # We'll evaluate constant-only first
        S0 = (0,)
        w0 = self._safe_solve(Gt[np.ix_(S0, S0)], bt[list(S0)], self.ridge)
        C0 = 0
        crit0, mse0, _ = self._score_subset(w0, Gv[np.ix_(S0, S0)], bv[list(S0)], yty_v, C0, 1)
        best.update({"crit": crit0, "mse_v": mse0, "S": S0, "w": w0, "C": C0, "k": 0})

        idxs = list(range(1, m))
        max_k = min(self.max_terms, len(idxs))

        # enumerate subsets up to max_k (non-constant)
        # Prefer smaller models when scores are close.
        for k in range(1, max_k + 1):
            for comb in itertools.combinations(idxs, k):
                S = (0,) + comb
                S_list = list(S)
                Gt_S = Gt[np.ix_(S_list, S_list)]
                bt_S = bt[S_list]
                w = self._safe_solve(Gt_S, bt_S, self.ridge)

                # complexity from terms + plus ops
                C_terms = 0
                for j in comb:
                    C_terms += self._term_complexity(comp_red[j])
                C_subset = C_terms + 2 * (len(S) - 1)

                crit, mse_v, _ = self._score_subset(
                    w,
                    Gv[np.ix_(S_list, S_list)],
                    bv[S_list],
                    yty_v,
                    C_subset,
                    len(S),
                )

                if crit < best["crit"] * 0.9995:
                    best.update({"crit": crit, "mse_v": mse_v, "S": S, "w": w, "C": C_subset, "k": k})
                elif crit <= best["crit"] * 1.0005:
                    # tie-break: lower complexity, fewer terms
                    if (C_subset < best["C"]) or (C_subset == best["C"] and k < best["k"]):
                        best.update({"crit": crit, "mse_v": mse_v, "S": S, "w": w, "C": C_subset, "k": k})

        # Refit on all data using selected subset
        keep2, Phi_all_red, names_red2, comp_red2 = self._select_features(Phi_full, y, names, comp_full)
        # Map best subset indices (in Phi_tr_red space) to original kept indices, then to Phi_all_red space
        # We'll instead rebuild using same keep from train selection, applied to full Phi, for stability.
        Phi_all_sel = Phi_full[:, keep]
        G_all = Phi_all_sel.T @ Phi_all_sel
        b_all = Phi_all_sel.T @ y

        S_best = best["S"]
        S_list = list(S_best)
        w_all = self._safe_solve(G_all[np.ix_(S_list, S_list)], b_all[S_list], self.ridge)

        # build final expression from selected names and coefficients
        sel_names = [names_red[i] for i in S_list]
        expr, C_expr = self._build_expression(sel_names, w_all)

        # predictions
        preds = Phi_all_sel[:, S_list] @ w_all
        preds = np.asarray(preds, dtype=np.float64)

        return {
            "expression": expr,
            "predictions": preds.tolist(),
            "details": {"complexity": int(C_expr)},
        }