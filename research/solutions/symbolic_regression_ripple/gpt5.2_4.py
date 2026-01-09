import numpy as np

def _safe_lstsq(A: np.ndarray, y: np.ndarray):
    try:
        coef, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
        return coef
    except Exception:
        # Fallback to ridge-regularized normal equations
        ATA = A.T @ A
        ATy = A.T @ y
        lam = 1e-10 * (np.trace(ATA) / max(1, ATA.shape[0]))
        try:
            coef = np.linalg.solve(ATA + lam * np.eye(ATA.shape[0]), ATy)
            return coef
        except Exception:
            return np.zeros(A.shape[1], dtype=float)

def _build_design(z_pows, arg, w, P, K):
    n = arg.shape[0]
    m = (P + 1) + 2 * (K + 1)
    A = np.empty((n, m), dtype=float)
    col = 0
    for k in range(P + 1):
        A[:, col] = z_pows[k]
        col += 1
    s = np.sin(w * arg)
    c = np.cos(w * arg)
    for k in range(K + 1):
        A[:, col] = z_pows[k] * s
        col += 1
    for k in range(K + 1):
        A[:, col] = z_pows[k] * c
        col += 1
    return A

def _mse(y, yhat):
    r = y - yhat
    return float(np.mean(r * r))

def _format_float(x: float) -> str:
    if not np.isfinite(x):
        x = 0.0
    if abs(x) < 1e-15:
        x = 0.0
    s = "{:.12g}".format(float(x))
    if s == "-0":
        s = "0"
    return s

def _pow_expr(base_expr: str, k: int) -> str:
    if k == 0:
        return "1"
    if k == 1:
        return base_expr
    return f"({base_expr})**{k}"

def _term_expr(coef: float, base_expr: str, k: int, trig: str, w: float, arg_expr: str) -> str:
    c = _format_float(coef)
    if trig is None:
        if k == 0:
            return c
        return f"({c})*{_pow_expr(base_expr, k)}"
    arg = f"({_format_float(w)})*({arg_expr})"
    fcall = f"{trig}({arg})"
    if k == 0:
        return f"({c})*{fcall}"
    return f"({c})*{_pow_expr(base_expr, k)}*{fcall}"

def _join_sum(terms):
    if not terms:
        return "0"
    expr = terms[0]
    for t in terms[1:]:
        expr = f"({expr}) + ({t})"
    return expr

def _linear_baseline(x1, x2, y):
    A = np.column_stack([x1, x2, np.ones_like(x1)])
    coef = _safe_lstsq(A, y)
    yhat = A @ coef
    return coef, yhat, _mse(y, yhat)

class Solution:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def solve(self, X: np.ndarray, y: np.ndarray) -> dict:
        X = np.asarray(X)
        y = np.asarray(y).reshape(-1)
        n = X.shape[0]
        if n == 0:
            return {"expression": "0", "predictions": [], "details": {"mse": 0.0}}

        x1 = X[:, 0].astype(float, copy=False)
        x2 = X[:, 1].astype(float, copy=False)
        r = x1 * x1 + x2 * x2
        t = np.sqrt(r)

        # Basic normalization safeguard
        y_std = float(np.std(y))
        if not np.isfinite(y_std) or y_std == 0.0:
            y_mean = float(np.mean(y)) if n else 0.0
            expr = _format_float(y_mean)
            return {"expression": expr, "predictions": (np.full(n, y_mean)).tolist(), "details": {"mse": _mse(y, np.full(n, y_mean))}}

        # Model settings (small to keep runtime/complexity low)
        P = int(self.kwargs.get("poly_degree", 3))
        K = int(self.kwargs.get("trig_poly_degree", 2))

        # Precompute powers up to max(P, K)
        max_pow = max(P, K)
        t_pows = [np.ones_like(t)]
        r_pows = [np.ones_like(r)]
        for k in range(1, max_pow + 1):
            t_pows.append(t_pows[-1] * t)
            r_pows.append(r_pows[-1] * r)

        # Candidate families: (z_pows, z_expr, arg, arg_expr, family_name)
        t_expr = "(x1**2 + x2**2)**0.5"
        r_expr = "(x1**2 + x2**2)"
        families = [
            (t_pows, t_expr, t, t_expr, "z=t,arg=t"),
            (t_pows, t_expr, r, r_expr, "z=t,arg=r"),
            (r_pows, r_expr, r, r_expr, "z=r,arg=r"),
            (r_pows, r_expr, t, t_expr, "z=r,arg=t"),
        ]

        # Baseline
        lin_coef, lin_pred, lin_mse = _linear_baseline(x1, x2, y)

        best = {
            "mse": lin_mse,
            "family": "linear",
            "w": 0.0,
            "coef": lin_coef,
            "P": None,
            "K": None,
            "pred": lin_pred,
            "expr": f"({_format_float(lin_coef[0])})*x1 + ({_format_float(lin_coef[1])})*x2 + ({_format_float(lin_coef[2])})",
            "n_terms": 3,
        }

        # Search each family with coarse-to-fine grid over w
        for z_pows, z_expr, arg, arg_expr, fam_name in families:
            arg_max = float(np.max(arg))
            if not np.isfinite(arg_max) or arg_max <= 0:
                continue

            w_min = max(0.05, 0.5 * np.pi / (arg_max + 1e-12))
            w_max = min(100.0, 60.0 * np.pi / (arg_max + 1e-12))
            if w_max <= w_min:
                continue

            # Coarse grid
            Wc = int(self.kwargs.get("w_grid_coarse", 72))
            wf = int(self.kwargs.get("w_grid_fine", 72))
            ws = np.linspace(w_min, w_max, Wc, dtype=float)
            step = (w_max - w_min) / max(1, (Wc - 1))

            best_local = {"mse": np.inf, "w": None, "coef": None, "A": None}
            for w in ws:
                A = _build_design(z_pows, arg, w, P, K)
                coef = _safe_lstsq(A, y)
                pred = A @ coef
                mse = _mse(y, pred)
                if mse < best_local["mse"]:
                    best_local = {"mse": mse, "w": float(w), "coef": coef, "A": A}

            if best_local["w"] is None or not np.isfinite(best_local["mse"]):
                continue

            # Fine grid around best
            w0 = best_local["w"]
            w_lo = max(w_min, w0 - 3.0 * step)
            w_hi = min(w_max, w0 + 3.0 * step)
            if w_hi > w_lo:
                ws2 = np.linspace(w_lo, w_hi, wf, dtype=float)
                for w in ws2:
                    A = _build_design(z_pows, arg, w, P, K)
                    coef = _safe_lstsq(A, y)
                    pred = A @ coef
                    mse = _mse(y, pred)
                    if mse < best_local["mse"]:
                        best_local = {"mse": mse, "w": float(w), "coef": coef, "A": A}

            # Optional pruning/refit to reduce complexity without harming fit too much
            A = best_local["A"]
            coef = best_local["coef"]
            if A is not None and coef is not None and np.all(np.isfinite(coef)):
                scales = np.sqrt(np.mean(A * A, axis=0))
                contrib = np.abs(coef) * (scales + 1e-18)
                thresh = max(1e-10 * float(np.max(contrib)), 1e-12 * y_std)
                keep = contrib > thresh
                keep[0] = True  # keep intercept
                if np.sum(keep) < A.shape[1]:
                    A2 = A[:, keep]
                    coef2 = _safe_lstsq(A2, y)
                    pred2 = A2 @ coef2
                    mse2 = _mse(y, pred2)
                    if np.isfinite(mse2) and mse2 <= best_local["mse"] * 1.001:
                        # Expand back to full coef with zeros
                        full = np.zeros_like(coef)
                        full[keep] = coef2
                        coef = full
                        best_local["coef"] = coef
                        best_local["mse"] = mse2

            mse = best_local["mse"]
            if np.isfinite(mse) and mse < best["mse"]:
                # Build expression from coef
                coef = best_local["coef"]
                w = best_local["w"]
                terms = []

                # Polynomial terms (k=0..P)
                for k in range(P + 1):
                    c = float(coef[k])
                    if abs(c) <= 1e-14 * (1.0 + y_std):
                        continue
                    terms.append(_term_expr(c, z_expr, k, None, w, arg_expr))

                # Trig terms
                offset = (P + 1)
                for k in range(K + 1):
                    c = float(coef[offset + k])
                    if abs(c) <= 1e-14 * (1.0 + y_std):
                        continue
                    terms.append(_term_expr(c, z_expr, k, "sin", w, arg_expr))
                offset += (K + 1)
                for k in range(K + 1):
                    c = float(coef[offset + k])
                    if abs(c) <= 1e-14 * (1.0 + y_std):
                        continue
                    terms.append(_term_expr(c, z_expr, k, "cos", w, arg_expr))

                expr = _join_sum(terms)
                # Predictions
                A_best = _build_design(z_pows, arg, w, P, K)
                pred = A_best @ best_local["coef"]

                best = {
                    "mse": float(mse),
                    "family": fam_name,
                    "w": float(w),
                    "coef": best_local["coef"],
                    "P": P,
                    "K": K,
                    "pred": pred,
                    "expr": expr,
                    "n_terms": len(terms),
                }

        return {
            "expression": best["expr"],
            "predictions": np.asarray(best["pred"], dtype=float).tolist(),
            "details": {
                "mse": float(best["mse"]),
                "family": best["family"],
                "w": float(best["w"]),
                "n_terms": int(best.get("n_terms", 0)),
            },
        }