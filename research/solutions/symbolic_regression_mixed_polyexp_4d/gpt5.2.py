import numpy as np

class Solution:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def solve(self, X: np.ndarray, y: np.ndarray) -> dict:
        X = np.asarray(X)
        y = np.asarray(y).reshape(-1)
        n, p = X.shape
        if p != 4:
            raise ValueError("Expected X shape (n, 4).")

        rng = np.random.default_rng(0)

        # If dataset is huge, fit on a subsample for speed/memory safety.
        max_fit_n = 30000
        if n > max_fit_n:
            fit_idx = rng.choice(n, size=max_fit_n, replace=False)
            X_fit = X[fit_idx]
            y_fit = y[fit_idx]
        else:
            X_fit = X
            y_fit = y

        n_fit = X_fit.shape[0]

        # Train/val split
        if n_fit >= 250:
            perm = rng.permutation(n_fit)
            n_tr = int(0.8 * n_fit)
            tr_idx = perm[:n_tr]
            va_idx = perm[n_tr:]
        else:
            tr_idx = np.arange(n_fit)
            va_idx = np.arange(n_fit)

        X_tr = X_fit[tr_idx]
        y_tr = y_fit[tr_idx]
        X_va = X_fit[va_idx]
        y_va = y_fit[va_idx]

        # Adaptive library size based on n to limit memory
        if n_fit > 60000:
            max_degree = 1
            use_q = ["x1**2", "x2**2", "x3**2", "x4**2", "(x1**2+x2**2+x3**2+x4**2)"]
            k_mults = (1.0, 2.0)
        elif n_fit > 20000:
            max_degree = 2
            use_q = ["x1**2", "x2**2", "x3**2", "x4**2", "(x1**2+x2**2+x3**2+x4**2)", "(x3**2+x4**2)"]
            k_mults = (0.5, 1.0, 2.0)
        else:
            max_degree = 3
            use_q = ["x1**2", "x2**2", "x3**2", "x4**2", "(x1**2+x2**2+x3**2+x4**2)", "(x1**2+x2**2)", "(x3**2+x4**2)", "(x1**2+x4**2)"]
            k_mults = (0.5, 1.0, 2.0, 4.0)

        x1_tr, x2_tr, x3_tr, x4_tr = X_tr[:, 0], X_tr[:, 1], X_tr[:, 2], X_tr[:, 3]
        x1_va, x2_va, x3_va, x4_va = X_va[:, 0], X_va[:, 1], X_va[:, 2], X_va[:, 3]

        def _safe_median(a):
            a = np.asarray(a)
            m = np.median(a)
            if not np.isfinite(m) or m <= 1e-12:
                return 1.0
            return float(m)

        def _format_float(v):
            if not np.isfinite(v):
                v = 0.0
            s = f"{float(v):.12g}"
            if s in ("-0", "-0.0"):
                s = "0"
            return s

        # Generate monomials up to max_degree
        # Exponents in graded lex order
        exps = [(0, 0, 0, 0)]
        for total in range(1, max_degree + 1):
            for e1 in range(total + 1):
                for e2 in range(total - e1 + 1):
                    for e3 in range(total - e1 - e2 + 1):
                        e4 = total - e1 - e2 - e3
                        exps.append((e1, e2, e3, e4))

        # Precompute powers for train/val
        def _powers(x):
            return [x, x * x, x * x * x]

        p1_tr, p2_tr, p3_tr, p4_tr = _powers(x1_tr), _powers(x2_tr), _powers(x3_tr), _powers(x4_tr)
        p1_va, p2_va, p3_va, p4_va = _powers(x1_va), _powers(x2_va), _powers(x3_va), _powers(x4_va)

        def _monomial_values_and_strs(exps_list, p1, p2, p3, p4):
            m = len(exps_list)
            vals = np.empty((p1[0].shape[0], m), dtype=np.float64)
            strs = []
            for j, (e1, e2, e3, e4) in enumerate(exps_list):
                v = 1.0
                parts = []
                if e1:
                    v = v * (p1[e1 - 1] if e1 <= 3 else (p1[0] ** e1))
                    parts.append("x1" if e1 == 1 else f"x1**{e1}")
                if e2:
                    v = v * (p2[e2 - 1] if e2 <= 3 else (p2[0] ** e2))
                    parts.append("x2" if e2 == 1 else f"x2**{e2}")
                if e3:
                    v = v * (p3[e3 - 1] if e3 <= 3 else (p3[0] ** e3))
                    parts.append("x3" if e3 == 1 else f"x3**{e3}")
                if e4:
                    v = v * (p4[e4 - 1] if e4 <= 3 else (p4[0] ** e4))
                    parts.append("x4" if e4 == 1 else f"x4**{e4}")
                vals[:, j] = v if isinstance(v, np.ndarray) else np.full(p1[0].shape[0], float(v), dtype=np.float64)
                strs.append("1" if not parts else "*".join(parts))
            return vals, strs

        mono_tr, mono_strs = _monomial_values_and_strs(exps, p1_tr, p2_tr, p3_tr, p4_tr)
        mono_va, _ = _monomial_values_and_strs(exps, p1_va, p2_va, p3_va, p4_va)

        # Build Q arrays for train/val from the selected expressions
        def _compute_Q(expr, x1, x2, x3, x4):
            if expr == "x1**2":
                return x1 * x1
            if expr == "x2**2":
                return x2 * x2
            if expr == "x3**2":
                return x3 * x3
            if expr == "x4**2":
                return x4 * x4
            if expr == "(x1**2+x2**2+x3**2+x4**2)":
                return x1 * x1 + x2 * x2 + x3 * x3 + x4 * x4
            if expr == "(x1**2+x2**2)":
                return x1 * x1 + x2 * x2
            if expr == "(x3**2+x4**2)":
                return x3 * x3 + x4 * x4
            if expr == "(x1**2+x4**2)":
                return x1 * x1 + x4 * x4
            # Fallback safe eval (should not happen)
            loc = {"x1": x1, "x2": x2, "x3": x3, "x4": x4}
            return eval(expr, {"__builtins__": {}}, loc)

        Q_tr_list = []
        Q_va_list = []
        for qexpr in use_q:
            Q_tr_list.append(_compute_Q(qexpr, x1_tr, x2_tr, x3_tr, x4_tr).astype(np.float64, copy=False))
            Q_va_list.append(_compute_Q(qexpr, x1_va, x2_va, x3_va, x4_va).astype(np.float64, copy=False))

        # Build list of g functions (values on train/val + sympy string)
        g_tr_list = [np.ones_like(y_tr, dtype=np.float64)]
        g_va_list = [np.ones_like(y_va, dtype=np.float64)]
        g_exprs = ["1"]

        # Add exp(-k*Q) for various Q and k
        seen = set()
        for qexpr, Q_tr, Q_va in zip(use_q, Q_tr_list, Q_va_list):
            med = _safe_median(Q_tr)
            base_k = 1.0 / med
            for mult in k_mults:
                k = float(base_k * mult)
                if not np.isfinite(k):
                    continue
                k = max(1e-4, min(50.0, k))
                key = (qexpr, round(k, 10))
                if key in seen:
                    continue
                seen.add(key)
                gtr = np.exp(-k * Q_tr)
                gva = np.exp(-k * Q_va)
                # Skip nearly-constant dampings
                if np.std(gtr) < 1e-10:
                    continue
                k_str = _format_float(k)
                g_str = f"exp(-({k_str})*({qexpr}))"
                g_tr_list.append(gtr)
                g_va_list.append(gva)
                g_exprs.append(g_str)

        # Assemble full feature matrices for train/val
        m_mono = mono_tr.shape[1]
        n_g = len(g_tr_list)
        d = m_mono * n_g

        # Avoid excessive memory
        est_bytes = (mono_tr.shape[0] + mono_va.shape[0]) * d * 8
        if est_bytes > 700_000_000:
            # Reduce: keep only g=1 and a couple of global dampings; and reduce degree if needed.
            # Rebuild with degree 2 and limited Qs.
            max_degree = min(max_degree, 2)
            exps2 = [(0, 0, 0, 0)]
            for total in range(1, max_degree + 1):
                for e1 in range(total + 1):
                    for e2 in range(total - e1 + 1):
                        for e3 in range(total - e1 - e2 + 1):
                            e4 = total - e1 - e2 - e3
                            exps2.append((e1, e2, e3, e4))
            mono_tr, mono_strs = _monomial_values_and_strs(exps2, p1_tr, p2_tr, p3_tr, p4_tr)
            mono_va, _ = _monomial_values_and_strs(exps2, p1_va, p2_va, p3_va, p4_va)
            m_mono = mono_tr.shape[1]

            use_q2 = ["(x1**2+x2**2+x3**2+x4**2)", "(x3**2+x4**2)"]
            Q_tr_list2 = [_compute_Q(q, x1_tr, x2_tr, x3_tr, x4_tr) for q in use_q2]
            Q_va_list2 = [_compute_Q(q, x1_va, x2_va, x3_va, x4_va) for q in use_q2]

            g_tr_list = [np.ones_like(y_tr, dtype=np.float64)]
            g_va_list = [np.ones_like(y_va, dtype=np.float64)]
            g_exprs = ["1"]
            seen = set()
            for qexpr, Q_tr, Q_va in zip(use_q2, Q_tr_list2, Q_va_list2):
                med = _safe_median(Q_tr)
                base_k = 1.0 / med
                for mult in (1.0, 2.0):
                    k = float(base_k * mult)
                    k = max(1e-4, min(50.0, k))
                    key = (qexpr, round(k, 10))
                    if key in seen:
                        continue
                    seen.add(key)
                    gtr = np.exp(-k * Q_tr)
                    gva = np.exp(-k * Q_va)
                    if np.std(gtr) < 1e-10:
                        continue
                    k_str = _format_float(k)
                    g_str = f"exp(-({k_str})*({qexpr}))"
                    g_tr_list.append(gtr)
                    g_va_list.append(gva)
                    g_exprs.append(g_str)
            n_g = len(g_tr_list)
            d = m_mono * n_g

        Phi_tr = np.empty((mono_tr.shape[0], d), dtype=np.float64)
        Phi_va = np.empty((mono_va.shape[0], d), dtype=np.float64)
        feat_strs = [""] * d

        col = 0
        for gi in range(n_g):
            gtr = g_tr_list[gi]
            gva = g_va_list[gi]
            gexpr = g_exprs[gi]
            # Fill block
            Phi_tr[:, col:col + m_mono] = mono_tr * gtr[:, None]
            Phi_va[:, col:col + m_mono] = mono_va * gva[:, None]
            if gexpr == "1":
                for mj, ms in enumerate(mono_strs):
                    feat_strs[col + mj] = ms
            else:
                for mj, ms in enumerate(mono_strs):
                    if ms == "1":
                        feat_strs[col + mj] = gexpr
                    else:
                        feat_strs[col + mj] = f"({ms})*({gexpr})"
            col += m_mono

        # OMP / forward selection
        def _ridge_solve(A, b, lam=1e-10):
            # Solve min ||Ax-b||^2 + lam||x||^2
            try:
                G = A.T @ A
                G.flat[::G.shape[0] + 1] += lam
                rhs = A.T @ b
                return np.linalg.solve(G, rhs)
            except Exception:
                return np.linalg.lstsq(A, b, rcond=None)[0]

        def _mse(a, b):
            r = a - b
            return float(np.mean(r * r))

        # Force include constant feature (monomial "1" with g="1"), which should be at index 0.
        force_idx = 0

        norms = np.sqrt(np.sum(Phi_tr * Phi_tr, axis=0)) + 1e-12

        # Some light pre-screening: remove near-zero norm columns
        valid = norms > 1e-10
        valid[force_idx] = True
        valid_idx = np.where(valid)[0]
        Phi_tr2 = Phi_tr[:, valid_idx]
        Phi_va2 = Phi_va[:, valid_idx]
        norms2 = norms[valid_idx]
        feat_strs2 = [feat_strs[i] for i in valid_idx]
        force_idx2 = int(np.where(valid_idx == force_idx)[0][0])

        # OMP path, choose best on validation
        max_terms = 12 if n_fit <= 20000 else (10 if n_fit <= 60000 else 8)
        if Phi_tr2.shape[1] < max_terms:
            max_terms = Phi_tr2.shape[1]

        active = [force_idx2]
        A = Phi_tr2[:, active]
        coef = _ridge_solve(A, y_tr, lam=1e-10)
        pred_va = Phi_va2[:, active] @ coef
        best_mse = _mse(pred_va, y_va)
        best_active = active.copy()
        best_coef = coef.copy()

        residual = y_tr - (A @ coef)

        for _ in range(1, max_terms):
            corr = (Phi_tr2.T @ residual)
            corr = np.abs(corr) / norms2
            corr[active] = -np.inf
            j = int(np.argmax(corr))
            if not np.isfinite(corr[j]) or corr[j] <= 1e-14:
                break
            active.append(j)

            A = Phi_tr2[:, active]
            coef = _ridge_solve(A, y_tr, lam=1e-10)
            residual = y_tr - (A @ coef)

            pred_va = Phi_va2[:, active] @ coef
            mse_va = _mse(pred_va, y_va)

            # Prefer lower MSE; tie-break by fewer terms
            if mse_va < best_mse * (1.0 - 1e-9) or (abs(mse_va - best_mse) <= 1e-12 and len(active) < len(best_active)):
                best_mse = mse_va
                best_active = active.copy()
                best_coef = coef.copy()

            # Early stopping if residual becomes tiny
            if float(np.mean(residual * residual)) < 1e-16 * max(1.0, float(np.mean(y_tr * y_tr))):
                break

        # Prune tiny coefficients
        if best_coef.size > 1:
            max_abs = float(np.max(np.abs(best_coef)))
            keep = np.abs(best_coef) >= max(1e-12, 1e-6 * max_abs)
            keep[0] = True  # keep intercept
            if not np.all(keep):
                best_active = [a for a, k in zip(best_active, keep) if k]
                best_coef = best_coef[keep]

        # Build final expression string
        terms = []
        for c, idx in zip(best_coef, best_active):
            if not np.isfinite(c):
                continue
            c = float(c)
            t = feat_strs2[idx]
            if t == "1":
                terms.append((c, "1"))
            else:
                terms.append((c, t))

        # If everything got pruned, fall back to mean
        if not terms:
            intercept = float(np.mean(y_fit))
            expression = _format_float(intercept)
            return {"expression": expression, "predictions": None, "details": {"complexity": 0}}

        # Combine into a readable expression
        expr_parts = []
        for i, (c, t) in enumerate(terms):
            if abs(c) < 1e-15:
                continue
            sign = "-" if c < 0 else "+"
            ac = abs(c)
            ac_str = _format_float(ac)

            if t == "1":
                part = ac_str
            else:
                if abs(ac - 1.0) < 1e-12:
                    part = f"({t})"
                else:
                    part = f"({ac_str})*({t})"

            if not expr_parts:
                expr_parts.append(f"-{part}" if c < 0 else part)
            else:
                expr_parts.append(f" {sign} {part}")

        expression = "".join(expr_parts)
        if expression.strip() == "":
            expression = "0"

        return {
            "expression": expression,
            "predictions": None,
            "details": {"complexity": int(len(terms))}
        }