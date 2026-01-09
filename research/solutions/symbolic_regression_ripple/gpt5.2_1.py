import numpy as np

class Solution:
    def __init__(self, **kwargs):
        self.random_state = kwargs.get("random_state", 42)

    @staticmethod
    def _fmt(x: float) -> str:
        if not np.isfinite(x):
            return "0"
        if abs(x) < 1e-15:
            return "0"
        return f"{float(x):.15g}"

    @staticmethod
    def _add_term(expr: str, coef: float, term_expr: str | None, tol: float) -> str:
        if not np.isfinite(coef) or abs(coef) <= tol:
            return expr
        sign = "-" if coef < 0 else "+"
        c = abs(coef)
        if term_expr is None:
            t = Solution._fmt(c)
        else:
            if abs(c - 1.0) <= 1e-12:
                t = term_expr
            else:
                t = f"{Solution._fmt(c)}*{term_expr}"
        if not expr:
            return f"-{t}" if coef < 0 else t
        return f"{expr} {sign} {t}"

    @staticmethod
    def _poly_str(c0: float, c1: float, c2: float, t_expr: str, t2_expr: str, tol: float) -> str:
        s = ""
        s = Solution._add_term(s, c0, None, tol)
        s = Solution._add_term(s, c1, t_expr, tol)
        s = Solution._add_term(s, c2, t2_expr, tol)
        return s if s else "0"

    @staticmethod
    def _join_top(parts: list[str]) -> str:
        expr = ""
        for p in parts:
            if not p or p == "0":
                continue
            p = p.strip()
            if not expr:
                expr = p
            else:
                if p.startswith("-"):
                    expr = f"{expr} - {p[1:].lstrip()}"
                else:
                    expr = f"{expr} + {p}"
        return expr if expr else "0"

    @staticmethod
    def _estimate_complexity(dp: int, ds: int, dc: int) -> int:
        m_base = dp + 1
        m_sin = 0 if ds < 0 else (ds + 1)
        m_cos = 0 if dc < 0 else (dc + 1)

        unary = (1 if m_sin > 0 else 0) + (1 if m_cos > 0 else 0)

        binary = 0
        if m_base > 0:
            binary += 2 * (m_base - 1)
        if m_sin > 0:
            binary += 2 * (m_sin - 1) + 2
        if m_cos > 0:
            binary += 2 * (m_cos - 1) + 2

        top_terms = (1 if m_base > 0 else 0) + (1 if m_sin > 0 else 0) + (1 if m_cos > 0 else 0)
        binary += max(top_terms - 1, 0)

        return int(2 * binary + unary)

    @staticmethod
    def _make_configs():
        configs = []
        for dp in (0, 1, 2):
            for ds in (-1, 0, 1, 2):
                for dc in (-1, 0, 1, 2):
                    if ds == -1 and dc == -1:
                        continue
                    idx = [0]
                    if dp >= 1:
                        idx.append(1)
                    if dp >= 2:
                        idx.append(2)

                    if ds >= 0:
                        idx.append(3)
                    if ds >= 1:
                        idx.append(5)
                    if ds >= 2:
                        idx.append(7)

                    if dc >= 0:
                        idx.append(4)
                    if dc >= 1:
                        idx.append(6)
                    if dc >= 2:
                        idx.append(8)

                    idx = tuple(sorted(set(idx)))
                    c_est = Solution._estimate_complexity(dp, ds, dc)
                    configs.append((dp, ds, dc, idx, c_est))
        return configs

    @staticmethod
    def _freq_grid(t: np.ndarray, n_lin: int = 120, n_log: int = 80):
        max_t = float(np.max(np.abs(t))) if t.size else 1.0
        max_t = max(max_t, 1e-9)

        w_min = max(0.1, 2.0 / max_t)
        w_max = min(500.0, 200.0 / max_t)
        if not (w_min < w_max):
            w_min, w_max = 0.1, 100.0

        ws_lin = np.linspace(w_min, w_max, n_lin, dtype=np.float64)
        if w_min > 0:
            ws_log = np.logspace(np.log10(w_min), np.log10(w_max), n_log, dtype=np.float64)
            ws = np.unique(np.concatenate([ws_lin, ws_log]))
        else:
            ws = np.unique(ws_lin)

        ws = ws[np.isfinite(ws) & (ws > 0)]
        return ws

    def solve(self, X: np.ndarray, y: np.ndarray) -> dict:
        X = np.asarray(X)
        y = np.asarray(y).reshape(-1)
        n = y.size
        if n == 0:
            return {"expression": "0", "predictions": [], "details": {}}

        x1 = X[:, 0].astype(np.float64, copy=False)
        x2 = X[:, 1].astype(np.float64, copy=False)

        y = y.astype(np.float64, copy=False)
        yTy = float(np.dot(y, y))
        var_y = float(np.var(y)) if n > 1 else float(yTy)

        configs = self._make_configs()

        r = x1 * x1 + x2 * x2
        t_variants = [
            ("r", r, "(x1**2 + x2**2)"),
            ("r2", r * r, "((x1**2 + x2**2)**2)"),
        ]

        ws_by_variant = {}
        for name, t, _ in t_variants:
            ws0 = self._freq_grid(t)
            ws_by_variant[name] = ws0

        def pass_best_mse(ws_map):
            best_mse = np.inf
            best_w_map = {}
            for name, t, _ in t_variants:
                ws = ws_map[name]
                t2 = t * t
                ones = np.ones_like(t)
                F = np.empty((n, 9), dtype=np.float64)
                F[:, 0] = ones
                F[:, 1] = t
                F[:, 2] = t2
                for w in ws:
                    s = np.sin(w * t)
                    c = np.cos(w * t)
                    F[:, 3] = s
                    F[:, 4] = c
                    F[:, 5] = t * s
                    F[:, 6] = t * c
                    F[:, 7] = t2 * s
                    F[:, 8] = t2 * c

                    G = F.T @ F
                    b = F.T @ y

                    diag_mean = float(np.mean(np.diag(G)))
                    lam = 1e-12 * (diag_mean + 1.0)

                    for dp, ds, dc, idx, _ in configs:
                        Gs = G[np.ix_(idx, idx)].copy()
                        bs = b[list(idx)]
                        if lam > 0:
                            Gs.flat[:: (Gs.shape[0] + 1)] += lam
                        try:
                            coef = np.linalg.solve(Gs, bs)
                        except np.linalg.LinAlgError:
                            coef, _, _, _ = np.linalg.lstsq(Gs, bs, rcond=None)

                        mse = (yTy - 2.0 * float(np.dot(coef, bs)) + float(coef @ (Gs @ coef))) / n
                        if mse < best_mse:
                            best_mse = mse
                            best_w_map[name] = float(w)
            return best_mse, best_w_map

        best_mse_1, best_w_map_1 = pass_best_mse(ws_by_variant)

        for name, t, _ in t_variants:
            w_best = best_w_map_1.get(name, None)
            if w_best is None or not np.isfinite(w_best) or w_best <= 0:
                continue
            w_lo = max(w_best * 0.85, 1e-6)
            w_hi = w_best * 1.15
            ws_ref = np.linspace(w_lo, w_hi, 140, dtype=np.float64)
            ws_extra = np.array([w_best, w_best * 0.5, w_best * 2.0], dtype=np.float64)
            ws = np.unique(np.concatenate([ws_by_variant[name], ws_ref, ws_extra]))
            ws = ws[np.isfinite(ws) & (ws > 0)]
            ws_by_variant[name] = ws

        best_mse, _ = pass_best_mse(ws_by_variant)

        tol_rel = 2e-4
        tol_abs = 1e-12 * (var_y + 1.0)
        mse_allow = best_mse * (1.0 + tol_rel) + tol_abs

        best = None  # (C_est, mse, name, w, k_full)
        for name, t, t_expr in t_variants:
            ws = ws_by_variant[name]
            t2 = t * t
            ones = np.ones_like(t)
            F = np.empty((n, 9), dtype=np.float64)
            F[:, 0] = ones
            F[:, 1] = t
            F[:, 2] = t2
            for w in ws:
                s = np.sin(w * t)
                c = np.cos(w * t)
                F[:, 3] = s
                F[:, 4] = c
                F[:, 5] = t * s
                F[:, 6] = t * c
                F[:, 7] = t2 * s
                F[:, 8] = t2 * c

                G = F.T @ F
                b = F.T @ y

                diag_mean = float(np.mean(np.diag(G)))
                lam = 1e-12 * (diag_mean + 1.0)

                for dp, ds, dc, idx, c_est in configs:
                    Gs = G[np.ix_(idx, idx)].copy()
                    bs = b[list(idx)]
                    if lam > 0:
                        Gs.flat[:: (Gs.shape[0] + 1)] += lam
                    try:
                        coef = np.linalg.solve(Gs, bs)
                    except np.linalg.LinAlgError:
                        coef, _, _, _ = np.linalg.lstsq(Gs, bs, rcond=None)

                    mse = (yTy - 2.0 * float(np.dot(coef, bs)) + float(coef @ (Gs @ coef))) / n
                    if not np.isfinite(mse) or mse > mse_allow:
                        continue
                    k_full = np.zeros(9, dtype=np.float64)
                    k_full[list(idx)] = coef
                    candidate = (c_est, float(mse), name, float(w), t_expr, k_full)
                    if best is None:
                        best = candidate
                    else:
                        if candidate[0] < best[0] - 1e-12:
                            best = candidate
                        elif abs(candidate[0] - best[0]) <= 1e-12 and candidate[1] < best[1]:
                            best = candidate

        if best is None:
            A = np.column_stack([x1, x2, np.ones_like(x1)])
            coef, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
            a, b0, c0 = coef
            expr = f"{self._fmt(a)}*x1 + {self._fmt(b0)}*x2 + {self._fmt(c0)}"
            preds = (a * x1 + b0 * x2 + c0).tolist()
            return {"expression": expr, "predictions": preds, "details": {"mse": float(np.mean((y - (a * x1 + b0 * x2 + c0)) ** 2))}}

        c_est, mse, name, w, t_expr, k = best
        if name == "r":
            t = r
        else:
            t = r * r
        t2 = t * t
        w = float(w)

        p0, p1, p2 = float(k[0]), float(k[1]), float(k[2])
        a0, b0c = float(k[3]), float(k[4])
        a1, b1c = float(k[5]), float(k[6])
        a2, b2c = float(k[7]), float(k[8])

        y_scale = float(np.std(y)) + 1e-12
        coef_tol = 1e-12 * (y_scale + 1.0)
        for ref in ("p0", "p1", "p2", "a0", "b0c", "a1", "b1c", "a2", "b2c"):
            pass

        def zsmall(v):
            return 0.0 if (not np.isfinite(v) or abs(v) <= coef_tol) else float(v)

        p0, p1, p2 = zsmall(p0), zsmall(p1), zsmall(p2)
        a0, a1, a2 = zsmall(a0), zsmall(a1), zsmall(a2)
        b0c, b1c, b2c = zsmall(b0c), zsmall(b1c), zsmall(b2c)

        w_str = self._fmt(w)
        t2_expr = f"({t_expr}**2)"
        arg = f"{w_str}*{t_expr}"

        base_poly = self._poly_str(p0, p1, p2, t_expr, t2_expr, tol=coef_tol)
        sin_poly = self._poly_str(a0, a1, a2, t_expr, t2_expr, tol=coef_tol)
        cos_poly = self._poly_str(b0c, b1c, b2c, t_expr, t2_expr, tol=coef_tol)

        parts = []
        if base_poly != "0":
            parts.append(base_poly)
        if sin_poly != "0":
            parts.append(f"({sin_poly})*sin({arg})")
        if cos_poly != "0":
            parts.append(f"({cos_poly})*cos({arg})")
        expression = self._join_top(parts)

        s = np.sin(w * t)
        c = np.cos(w * t)
        preds = (
            (p0 + p1 * t + p2 * t2)
            + (a0 + a1 * t + a2 * t2) * s
            + (b0c + b1c * t + b2c * t2) * c
        )

        return {
            "expression": expression,
            "predictions": preds.tolist(),
            "details": {"mse": float(np.mean((y - preds) ** 2)), "complexity_est": int(c_est)},
        }