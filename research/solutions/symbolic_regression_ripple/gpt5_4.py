import numpy as np

class Solution:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def _fmt_num(self, v):
        if not np.isfinite(v):
            v = 0.0
        return f"{float(v):.12g}"

    def _poly_str(self, coefs, var_expr, tol):
        terms = []
        for i, c in enumerate(coefs):
            if abs(c) <= tol:
                continue
            c_str = f"({self._fmt_num(c)})"
            if i == 0:
                terms.append(c_str)
            elif i == 1:
                terms.append(f"{c_str}*{var_expr}")
            else:
                terms.append(f"{c_str}*({var_expr}**{i})")
        if not terms:
            return "0"
        return " + ".join(terms)

    def _build_expression(self, w, s_expr, a_coefs, b_coefs, c_coefs, r_expr, tol):
        parts = []
        if a_coefs is not None and any(abs(c) > tol for c in a_coefs):
            A_poly = self._poly_str(a_coefs, r_expr, tol)
            parts.append(f"({A_poly})*sin({self._fmt_num(w)}*{s_expr})")
        if b_coefs is not None and any(abs(c) > tol for c in b_coefs):
            B_poly = self._poly_str(b_coefs, r_expr, tol)
            parts.append(f"({B_poly})*cos({self._fmt_num(w)}*{s_expr})")
        if c_coefs is not None and any(abs(c) > tol for c in c_coefs):
            C_poly = self._poly_str(c_coefs, r_expr, tol)
            parts.append(f"{C_poly}")
        if not parts:
            return "0"
        return " + ".join(parts)

    def _fit_best_radial_model(self, X, y):
        n = len(y)
        x1 = X[:, 0]
        x2 = X[:, 1]
        R = np.sqrt(x1 * x1 + x2 * x2)
        RR = x1 * x1 + x2 * x2

        # Candidates for the sinusoidal argument
        s_candidates = [
            ("r", R, "(x1**2 + x2**2)**0.5"),
            ("rr", RR, "(x1**2 + x2**2)")
        ]

        # Degree options for amplitude polynomials
        deg_options = [0, 1, 2]
        # Numerical stability tolerance
        tol = 1e-12

        best = {
            "score": np.inf,
            "pred": None,
            "w": None,
            "s_expr": None,
            "a_deg": None,
            "b_deg": None,
            "c_deg": None,
            "a_coefs": None,
            "b_coefs": None,
            "c_coefs": None,
        }

        ones = np.ones_like(R)
        powR = [ones, R, R * R]

        # W grids for each s
        # These ranges are chosen to cover plausible frequencies without heavy computation
        s_grids = {
            "r": np.linspace(0.5, 30.0, 120),
            "rr": np.linspace(0.02, 3.0, 80),
        }

        y = y.astype(float)
        var_y = np.var(y)
        # Pre-check: if y is almost constant, short circuit
        if not np.isfinite(var_y) or var_y < 1e-20:
            const = float(np.mean(y))
            expr = self._fmt_num(const)
            return expr, np.full_like(y, const, dtype=float)

        for s_name, s_vals, s_expr in s_candidates:
            w_grid = s_grids[s_name]
            # Precompute quantiles to exclude extremes if needed (not used in this approach)
            for w in w_grid:
                S = np.sin(w * s_vals)
                C = np.cos(w * s_vals)

                # Precompute products with R powers
                RS = [powR[i] * S for i in range(3)]
                RC = [powR[i] * C for i in range(3)]

                for a_deg in deg_options:
                    for b_deg in deg_options:
                        for c_deg in deg_options:
                            # Build design matrix
                            feats = []
                            # A(r) * sin(w*s)
                            for i in range(a_deg + 1):
                                feats.append(RS[i])
                            # B(r) * cos(w*s)
                            for i in range(b_deg + 1):
                                feats.append(RC[i])
                            # C(r)
                            for i in range(c_deg + 1):
                                feats.append(powR[i])

                            Xf = np.column_stack(feats)
                            k = Xf.shape[1]
                            # Guard against tiny sample vs parameter count
                            try:
                                coefs, _, _, _ = np.linalg.lstsq(Xf, y, rcond=None)
                                pred = Xf @ coefs
                            except Exception:
                                continue

                            resid = y - pred
                            SSE = float(np.dot(resid, resid))
                            # AICc with small epsilon to avoid log(0)
                            if n > k + 1:
                                AICc = n * np.log(SSE / n + 1e-300) + 2 * k + (2 * k * (k + 1)) / (n - k - 1)
                            else:
                                AICc = n * np.log(SSE / n + 1e-300) + 2 * k + 1e6

                            if AICc < best["score"]:
                                idx = 0
                                a_len = a_deg + 1
                                b_len = b_deg + 1
                                c_len = c_deg + 1
                                a_coefs = coefs[idx:idx + a_len]
                                idx += a_len
                                b_coefs = coefs[idx:idx + b_len]
                                idx += b_len
                                c_coefs = coefs[idx:idx + c_len]

                                best.update({
                                    "score": AICc,
                                    "pred": pred,
                                    "w": w,
                                    "s_expr": s_expr,
                                    "a_deg": a_deg,
                                    "b_deg": b_deg,
                                    "c_deg": c_deg,
                                    "a_coefs": a_coefs,
                                    "b_coefs": b_coefs,
                                    "c_coefs": c_coefs,
                                })

        # If nothing found (shouldn't happen), return baseline
        if best["w"] is None:
            # Baseline linear fit
            A = np.column_stack([X[:, 0], X[:, 1], np.ones(len(y))])
            coeffs, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
            pred = A @ coeffs
            a, b, c = coeffs
            expr = f"{self._fmt_num(a)}*x1 + {self._fmt_num(b)}*x2 + {self._fmt_num(c)}"
            return expr, pred

        # Prune tiny coefficients to simplify expression
        all_coefs = np.concatenate([best["a_coefs"], best["b_coefs"], best["c_coefs"]])
        scale = max(np.max(np.abs(all_coefs)), 1.0)
        prune_tol = 1e-6 * scale

        a_coefs = np.array([c if abs(c) > prune_tol else 0.0 for c in best["a_coefs"]], dtype=float)
        b_coefs = np.array([c if abs(c) > prune_tol else 0.0 for c in best["b_coefs"]], dtype=float)
        c_coefs = np.array([c if abs(c) > prune_tol else 0.0 for c in best["c_coefs"]], dtype=float)

        r_expr = "(x1**2 + x2**2)**0.5"
        expr = self._build_expression(best["w"], best["s_expr"], a_coefs, b_coefs, c_coefs, r_expr, tol=1e-15)

        return expr, best["pred"]

    def solve(self, X: np.ndarray, y: np.ndarray) -> dict:
        # Fit radial-based symbolic model
        expression, predictions = self._fit_best_radial_model(X, y)
        return {
            "expression": expression,
            "predictions": predictions.tolist(),
            "details": {}
        }