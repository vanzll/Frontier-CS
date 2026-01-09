import numpy as np

def _fmt_coef(c: float) -> str:
    if not np.isfinite(c) or abs(c) < 1e-15:
        c = 0.0
    s = format(float(c), ".12g")
    if s == "-0":
        s = "0"
    return s

def _monomial_expr(exps):
    vars_ = ("x1", "x2", "x3", "x4")
    parts = []
    for v, e in zip(vars_, exps):
        if e == 0:
            continue
        if e == 1:
            parts.append(v)
        else:
            parts.append(f"{v}**{e}")
    if not parts:
        return "1"
    return "*".join(parts)

def _format_sum(terms, coef_eps=1e-12):
    kept = []
    for coef, mon in terms:
        if not np.isfinite(coef):
            continue
        if abs(coef) <= coef_eps:
            continue
        kept.append((float(coef), mon))
    if not kept:
        return "0"

    def term_str(c, mon_expr, first=False):
        is_const = (mon_expr == "1")
        if is_const:
            s = _fmt_coef(c)
            return s
        if abs(c - 1.0) < 1e-12:
            s = f"({mon_expr})"
        elif abs(c + 1.0) < 1e-12:
            s = f"-({mon_expr})"
        else:
            s = f"{_fmt_coef(c)}*({mon_expr})"
        return s

    out = ""
    for i, (c, mon) in enumerate(kept):
        if i == 0:
            out = term_str(c, mon, first=True)
        else:
            if c >= 0:
                out += " + " + term_str(c, mon)
            else:
                out += " - " + term_str(-c, mon)
    return out

def _ridge_solve(A, y, lam=1e-10):
    m = A.shape[1]
    AtA = A.T @ A
    if lam > 0:
        AtA = AtA + lam * np.eye(m, dtype=A.dtype)
    Aty = A.T @ y
    try:
        return np.linalg.solve(AtA, Aty)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(A, y, rcond=None)[0]

def _omp_select(A, y, max_terms=16, fixed_idx=(0,), min_improve=1e-12, min_corr=1e-12):
    n, m = A.shape
    col_norms = np.linalg.norm(A, axis=0)
    col_norms = np.where(col_norms < 1e-12, 1.0, col_norms)
    An = A / col_norms

    selected = list(dict.fromkeys(int(i) for i in fixed_idx if 0 <= int(i) < m))
    selected_mask = np.zeros(m, dtype=bool)
    selected_mask[selected] = True

    if selected:
        beta = np.linalg.lstsq(An[:, selected], y, rcond=None)[0]
        resid = y - An[:, selected] @ beta
    else:
        resid = y.copy()

    prev_mse = float(np.mean(resid * resid))
    y_scale = float(np.linalg.norm(y) + 1e-12)

    for _ in range(max_terms - len(selected)):
        corr = An.T @ resid
        corr[selected_mask] = 0.0
        idx = int(np.argmax(np.abs(corr)))
        cval = float(abs(corr[idx]))
        if cval < min_corr * y_scale:
            break

        selected.append(idx)
        selected_mask[idx] = True

        beta = np.linalg.lstsq(An[:, selected], y, rcond=None)[0]
        resid = y - An[:, selected] @ beta
        mse = float(np.mean(resid * resid))

        if prev_mse - mse < min_improve * max(1.0, prev_mse):
            selected.pop()
            selected_mask[idx] = False
            break
        prev_mse = mse

    return selected, col_norms

def _build_candidates(X):
    x1 = X[:, 0].astype(np.float64, copy=False)
    x2 = X[:, 1].astype(np.float64, copy=False)
    x3 = X[:, 2].astype(np.float64, copy=False)
    x4 = X[:, 3].astype(np.float64, copy=False)

    n = X.shape[0]
    ones = np.ones(n, dtype=np.float64)

    xs = [x1, x2, x3, x4]
    max_pow = 4
    pows = []
    for x in xs:
        pw = [ones, x]
        for k in range(2, max_pow + 1):
            pw.append(pw[-1] * x)
        pows.append(pw)

    sumsq = x1 * x1 + x2 * x2 + x3 * x3 + x4 * x4

    terms_expr = []
    terms_val = []
    terms_mon = []
    terms_exp = []

    # Constant
    terms_expr.append("1")
    terms_val.append(ones)
    terms_mon.append("1")
    terms_exp.append(None)

    # Monomials up to degree 3
    mon_exps = []
    for d in range(1, 4):
        for e1 in range(d + 1):
            for e2 in range(d - e1 + 1):
                for e3 in range(d - e1 - e2 + 1):
                    e4 = d - e1 - e2 - e3
                    mon_exps.append((e1, e2, e3, e4))

    # Add degree-4 interaction term
    mon_exps.append((1, 1, 1, 1))

    mon_vals = {}
    mon_degs = {}
    for exps in mon_exps:
        e1, e2, e3, e4 = exps
        val = pows[0][e1] * pows[1][e2] * pows[2][e3] * pows[3][e4]
        expr = _monomial_expr(exps)
        mon_vals[expr] = val
        mon_degs[expr] = e1 + e2 + e3 + e4

    # Add base polynomial terms
    for expr, val in mon_vals.items():
        terms_expr.append(expr)
        terms_val.append(val)
        terms_mon.append(expr)
        terms_exp.append(None)

    # Global exp terms
    global_ks = (0.5, 1.0, 2.0)
    global_exp_vals = {}
    global_exp_exprs = {}
    for k in global_ks:
        k_str = _fmt_coef(k)
        exp_expr = f"exp(-{k_str}*(x1**2 + x2**2 + x3**2 + x4**2))"
        ev = np.exp(-k * sumsq)
        global_exp_vals[k] = ev
        global_exp_exprs[k] = exp_expr
        terms_expr.append(exp_expr)
        terms_val.append(ev)
        terms_mon.append("1")
        terms_exp.append(exp_expr)

    # Per-variable exp terms
    per_ks = (0.5, 1.0, 2.0)
    for i, x in enumerate(xs, start=1):
        xx = x * x
        for k in per_ks:
            k_str = _fmt_coef(k)
            exp_expr = f"exp(-{k_str}*x{i}**2)"
            ev = np.exp(-k * xx)
            terms_expr.append(exp_expr)
            terms_val.append(ev)
            terms_mon.append("1")
            terms_exp.append(exp_expr)

    # Products: low-degree monomials times global exp (to model poly * gaussian damping)
    prod_ks = (0.5, 1.0)
    for k in prod_ks:
        exp_expr = global_exp_exprs[k]
        ev = global_exp_vals[k]
        for mon_expr, mon_val in mon_vals.items():
            d = mon_degs.get(mon_expr, 0)
            if d <= 2 and mon_expr != "1":
                expr = f"({mon_expr})*({exp_expr})"
                terms_expr.append(expr)
                terms_val.append(mon_val * ev)
                terms_mon.append(mon_expr)
                terms_exp.append(exp_expr)

    A = np.column_stack(terms_val).astype(np.float64, copy=False)
    return A, terms_expr, terms_mon, terms_exp

def _baseline_linear(X, y):
    x1, x2, x3, x4 = X[:, 0], X[:, 1], X[:, 2], X[:, 3]
    A = np.column_stack([x1, x2, x3, x4, np.ones_like(x1)])
    coef = np.linalg.lstsq(A, y, rcond=None)[0]
    a, b, c, d, e = [float(v) for v in coef]
    expr = (
        f"{_fmt_coef(a)}*x1 + {_fmt_coef(b)}*x2 + {_fmt_coef(c)}*x3 + "
        f"{_fmt_coef(d)}*x4 + {_fmt_coef(e)}"
    )
    pred = A @ coef
    return expr, pred

class Solution:
    def __init__(self, **kwargs):
        self.max_terms = int(kwargs.get("max_terms", 16))
        self.lam = float(kwargs.get("lam", 1e-10))
        self.coef_prune = float(kwargs.get("coef_prune", 1e-10))

    def solve(self, X: np.ndarray, y: np.ndarray) -> dict:
        X = np.asarray(X)
        y = np.asarray(y).astype(np.float64, copy=False).ravel()

        if X.ndim != 2 or X.shape[1] != 4 or y.ndim != 1 or X.shape[0] != y.shape[0]:
            expr, pred = _baseline_linear(X, y)
            return {"expression": expr, "predictions": pred.tolist(), "details": {}}

        try:
            A, exprs, mon_exprs, exp_factors = _build_candidates(X)

            selected, col_norms = _omp_select(
                A,
                y,
                max_terms=self.max_terms,
                fixed_idx=(0,),
                min_improve=1e-12,
                min_corr=1e-12,
            )

            # Refit on original columns with ridge
            A_sel = A[:, selected]
            beta = _ridge_solve(A_sel, y, lam=self.lam)

            # Prune small coefficients (keep constant term if present)
            beta_abs = np.abs(beta)
            maxb = float(beta_abs.max()) if beta_abs.size else 0.0
            keep_mask = beta_abs >= (self.coef_prune * max(1.0, maxb))
            if 0 in selected:
                keep_mask[selected.index(0)] = True

            if not np.all(keep_mask):
                selected2 = [idx for idx, k in zip(selected, keep_mask) if k]
                if selected2:
                    A_sel2 = A[:, selected2]
                    beta2 = _ridge_solve(A_sel2, y, lam=self.lam)
                    selected, beta = selected2, beta2
                    A_sel = A_sel2

            # Build grouped expression by factoring common exp(...)
            groups = {}
            for coef, idx in zip(beta, selected):
                exp_fac = exp_factors[idx]
                mon = mon_exprs[idx]
                if exp_fac not in groups:
                    groups[exp_fac] = []
                groups[exp_fac].append((float(coef), mon))

            group_terms = []
            for exp_fac, tlist in groups.items():
                inner = _format_sum(tlist, coef_eps=self.coef_prune * max(1.0, float(np.max(np.abs(beta))) if beta.size else 1.0))
                if inner == "0":
                    continue
                if exp_fac is None:
                    group_terms.append(inner)
                else:
                    if inner == "1":
                        group_terms.append(exp_fac)
                    elif inner == "-1":
                        group_terms.append(f"-({exp_fac})")
                    else:
                        group_terms.append(f"({inner})*({exp_fac})")

            if not group_terms:
                expression = "0"
            elif len(group_terms) == 1:
                expression = group_terms[0]
            else:
                expression = " + ".join(f"({t})" for t in group_terms)

            pred = A_sel @ beta
            mse = float(np.mean((y - pred) ** 2)) if y.size else 0.0

            details = {
                "n_terms": int(len(selected)),
                "mse": mse,
            }

            return {
                "expression": expression,
                "predictions": pred.tolist(),
                "details": details,
            }
        except Exception:
            expr, pred = _baseline_linear(X, y)
            return {"expression": expr, "predictions": pred.tolist(), "details": {}}