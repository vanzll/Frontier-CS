import numpy as np

def _gen_exponent_tuples(nvars: int, degree: int):
    out = []
    exp = [0] * nvars

    def rec(i, remaining):
        if i == nvars - 1:
            exp[i] = remaining
            out.append(tuple(exp))
            return
        for v in range(remaining + 1):
            exp[i] = v
            rec(i + 1, remaining - v)

    rec(0, degree)
    return out


def _build_monomials(X, degree_max: int):
    n, d = X.shape
    powers = [[None] * (degree_max + 1) for _ in range(d)]
    for j in range(d):
        powers[j][0] = np.ones(n, dtype=np.float64)
        if degree_max >= 1:
            powers[j][1] = X[:, j].astype(np.float64, copy=False)
        for e in range(2, degree_max + 1):
            powers[j][e] = powers[j][e - 1] * powers[j][1]

    exps = []
    for deg in range(degree_max + 1):
        exps.extend(_gen_exponent_tuples(d, deg))

    mon_arrays = []
    mon_strings = []
    mon_degrees = []

    varnames = ["x1", "x2", "x3", "x4"]
    for e in exps:
        deg = sum(e)
        arr = np.ones(n, dtype=np.float64)
        parts = []
        for j, ej in enumerate(e):
            if ej == 0:
                continue
            arr = arr * powers[j][ej]
            if ej == 1:
                parts.append(varnames[j])
            else:
                parts.append(f"{varnames[j]}**{ej}")
        s = "1" if not parts else "*".join(parts)
        mon_arrays.append(arr)
        mon_strings.append(s)
        mon_degrees.append(deg)

    return mon_arrays, mon_strings, mon_degrees


def _build_dampings(X, k_values):
    x1, x2, x3, x4 = (X[:, 0].astype(np.float64, copy=False),
                      X[:, 1].astype(np.float64, copy=False),
                      X[:, 2].astype(np.float64, copy=False),
                      X[:, 3].astype(np.float64, copy=False))
    xs = [x1, x2, x3, x4]
    x2s = [x * x for x in xs]
    varnames = ["x1", "x2", "x3", "x4"]

    subsets = []
    for i in range(4):
        subsets.append((i,))
    for i in range(4):
        for j in range(i + 1, 4):
            subsets.append((i, j))
    subsets.append((0, 1, 2, 3))

    damp_arrays = []
    damp_strings = []
    for subset in subsets:
        sumsq = np.zeros_like(x1, dtype=np.float64)
        parts = []
        for idx in subset:
            sumsq = sumsq + x2s[idx]
            parts.append(f"{varnames[idx]}**2")
        sumsq_str = " + ".join(parts)
        for k in k_values:
            arr = np.exp(-k * sumsq)
            s = f"exp(-{k:.12g}*({sumsq_str}))"
            damp_arrays.append(arr)
            damp_strings.append(s)
    return damp_arrays, damp_strings


def _omp(F, y, names, max_terms=8, min_improve=1e-12):
    n, m = F.shape
    y = y.astype(np.float64, copy=False)

    mu = F.mean(axis=0)
    Fc = F - mu
    norms = np.sqrt(np.sum(Fc * Fc, axis=0))
    valid = norms > 1e-12
    if not np.any(valid):
        intercept = float(np.mean(y))
        return [], np.array([intercept], dtype=np.float64), np.full(n, intercept, dtype=np.float64)

    idx_map = np.nonzero(valid)[0]
    Fs = Fc[:, valid] / norms[valid]
    names_v = [names[i] for i in idx_map.tolist()]
    F_v = F[:, valid]

    selected = []
    selected_mask = np.zeros(Fs.shape[1], dtype=bool)

    A0 = np.ones((n, 1), dtype=np.float64)
    coef0, _, _, _ = np.linalg.lstsq(A0, y, rcond=None)
    pred = A0 @ coef0
    resid = y - pred
    best_mse = float(np.mean(resid * resid))

    for _ in range(min(max_terms, Fs.shape[1])):
        corr = np.abs(Fs.T @ resid)
        corr[selected_mask] = -1.0
        j = int(np.argmax(corr))
        if corr[j] <= 0:
            break

        selected.append(j)
        selected_mask[j] = True

        A = np.column_stack([np.ones(n, dtype=np.float64), F_v[:, selected]])
        coef, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
        pred_new = A @ coef
        resid_new = y - pred_new
        mse_new = float(np.mean(resid_new * resid_new))

        if best_mse - mse_new < min_improve * max(1.0, best_mse):
            selected.pop()
            break

        best_mse = mse_new
        resid = resid_new
        pred = pred_new

        if best_mse < 1e-24:
            break

    if selected:
        A = np.column_stack([np.ones(n, dtype=np.float64), F_v[:, selected]])
        coef, _, _, _ = np.linalg.lstsq(A, y, rcond=None)

        coefs_terms = coef[1:]
        maxabs = float(np.max(np.abs(coefs_terms))) if coefs_terms.size else 0.0
        if maxabs > 0:
            keep = np.abs(coefs_terms) >= (1e-10 * maxabs)
        else:
            keep = np.ones_like(coefs_terms, dtype=bool)

        if not np.all(keep):
            selected = [s for s, k in zip(selected, keep.tolist()) if k]
            if selected:
                A = np.column_stack([np.ones(n, dtype=np.float64), F_v[:, selected]])
                coef, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
                pred = A @ coef
            else:
                A = np.ones((n, 1), dtype=np.float64)
                coef, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
                pred = A @ coef
    else:
        coef = coef0
        pred = A0 @ coef

    selected_global = [int(idx_map[s]) for s in selected]
    return selected_global, coef.astype(np.float64), pred.astype(np.float64)


def _format_float(c):
    if not np.isfinite(c):
        return "0.0"
    if abs(c) < 5e-15:
        c = 0.0
    s = f"{c:.12g}"
    if s == "-0":
        s = "0"
    return s


def _build_expression(intercept, terms, coefs):
    expr = _format_float(intercept)
    for term, c in zip(terms, coefs):
        if not np.isfinite(c) or abs(c) < 5e-15:
            continue
        if c >= 0:
            expr += f" + ({_format_float(c)})*({term})"
        else:
            expr += f" - ({_format_float(-c)})*({term})"
    return expr


class Solution:
    def __init__(self, **kwargs):
        self.max_terms = int(kwargs.get("max_terms", 8))
        self.k_values = kwargs.get("k_values", [0.5, 1.0, 2.0])
        self.degree_max = int(kwargs.get("degree_max", 3))

    def solve(self, X: np.ndarray, y: np.ndarray) -> dict:
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).reshape(-1)

        n, d = X.shape
        if d != 4 or n == 0:
            return {"expression": "0.0", "predictions": [0.0] * int(n), "details": {}}

        try:
            mon_arrs, mon_strs, mon_degs = _build_monomials(X, self.degree_max)
            deg2_idx = [i for i, deg in enumerate(mon_degs) if deg <= 2]
            deg3_nonconst_idx = [i for i, deg in enumerate(mon_degs) if 1 <= deg <= 3]

            damp_arrs, damp_strs = _build_dampings(X, self.k_values)

            cols = []
            names = []

            for i in deg3_nonconst_idx:
                cols.append(mon_arrs[i])
                names.append(mon_strs[i])

            for darr, dstr in zip(damp_arrs, damp_strs):
                for i in deg2_idx:
                    cols.append(mon_arrs[i] * darr)
                    if mon_degs[i] == 0:
                        names.append(dstr)
                    else:
                        names.append(f"({mon_strs[i]})*({dstr})")

            F = np.column_stack(cols).astype(np.float64, copy=False)

            selected_idx, coef, pred = _omp(F, y, names, max_terms=self.max_terms)

            intercept = float(coef[0]) if coef.size >= 1 else float(np.mean(y))
            if selected_idx:
                terms = [names[i] for i in selected_idx]
                term_coefs = coef[1:1 + len(selected_idx)]
            else:
                terms = []
                term_coefs = np.array([], dtype=np.float64)

            expression = _build_expression(intercept, terms, term_coefs)

            return {
                "expression": expression,
                "predictions": pred.tolist(),
                "details": {}
            }
        except Exception:
            x1, x2, x3, x4 = X[:, 0], X[:, 1], X[:, 2], X[:, 3]
            A = np.column_stack([x1, x2, x3, x4, np.ones_like(x1)])
            coef, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
            a, b, c, d, e = coef.tolist()
            expression = f"({_format_float(a)})*x1 + ({_format_float(b)})*x2 + ({_format_float(c)})*x3 + ({_format_float(d)})*x4 + ({_format_float(e)})"
            pred = a * x1 + b * x2 + c * x3 + d * x4 + e
            return {
                "expression": expression,
                "predictions": pred.tolist(),
                "details": {}
            }