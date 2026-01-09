import numpy as np

def _float_to_str(v):
    return "{:.12g}".format(float(v))

def _kmeans_2d(X, K=8, max_iter=12, seed=0):
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    if n == 0:
        return np.zeros((K, 2))
    if K > n:
        K = n
    idx = rng.choice(n, size=K, replace=False)
    centers = X[idx].copy()
    for _ in range(max_iter):
        # Assign
        d2 = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        labels = np.argmin(d2, axis=1)
        # Update
        new_centers = centers.copy()
        for k in range(K):
            mask = labels == k
            if np.any(mask):
                new_centers[k] = X[mask].mean(axis=0)
            else:
                # Reinitialize empty clusters to random data points
                new_centers[k] = X[rng.integers(0, n)]
        if np.allclose(new_centers, centers):
            break
        centers = new_centers
    return centers

def _cluster_stds(X, centers):
    # Assign points to nearest center and compute per-cluster stds
    n = X.shape[0]
    K = centers.shape[0]
    d2 = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    labels = np.argmin(d2, axis=1)
    stds = np.zeros_like(centers)
    global_std = X.std(axis=0) + 1e-12
    for k in range(K):
        mask = labels == k
        if np.sum(mask) >= 3:
            stds[k] = X[mask].std(axis=0) + 1e-12
        else:
            stds[k] = global_std
    return stds

def _build_terms_and_features(X, max_centers=12, seed=0):
    n = X.shape[0]
    x1 = X[:, 0]
    x2 = X[:, 1]
    # Determine number of centers based on n
    K = min(max_centers, max(5, int(np.sqrt(max(n, 1)) / 3) + 4))
    centers = _kmeans_2d(X, K=K, max_iter=15, seed=seed)
    stds = _cluster_stds(X, centers)
    # Width scales
    scales = np.array([0.7, 1.0, 1.6])
    terms = []
    feats = []

    # Polynomial terms up to degree 3 (excluding constant)
    poly_degrees = []
    for d in range(1, 4):
        for p in range(d + 1):
            q = d - p
            poly_degrees.append((p, q))
    for (p, q) in poly_degrees:
        terms.append({'type': 'poly', 'p': p, 'q': q})
        col = (x1 ** p) * (x2 ** q)
        feats.append(col)

    # Gaussian-like terms: for each center and scale, generate
    # (0,0), (1,0), (0,1), (1,1) polynomial multipliers on (x - c)
    for k in range(K):
        c1, c2 = centers[k]
        s1, s2 = stds[k]
        s1 = float(s1)
        s2 = float(s2)
        # Base alphas
        # Using Gaussian: exp(- (a1*(x1 - c1)^2 + a2*(x2 - c2)^2))
        # a = 1 / (2 * (scale * std)^2)
        base_a1 = 1.0 / (2.0 * (s1 ** 2) + 1e-12)
        base_a2 = 1.0 / (2.0 * (s2 ** 2) + 1e-12)
        for sc in scales:
            a1 = base_a1 / (sc ** 2)
            a2 = base_a2 / (sc ** 2)
            dx1 = x1 - c1
            dx2 = x2 - c2
            g = np.exp(- (a1 * (dx1 ** 2) + a2 * (dx2 ** 2)))
            # (0,0)
            terms.append({'type': 'gp', 'c1': c1, 'c2': c2, 'a1': float(a1), 'a2': float(a2), 'p': 0, 'q': 0})
            feats.append(g)
            # (1,0)
            terms.append({'type': 'gp', 'c1': c1, 'c2': c2, 'a1': float(a1), 'a2': float(a2), 'p': 1, 'q': 0})
            feats.append(dx1 * g)
            # (0,1)
            terms.append({'type': 'gp', 'c1': c1, 'c2': c2, 'a1': float(a1), 'a2': float(a2), 'p': 0, 'q': 1})
            feats.append(dx2 * g)
            # (1,1)
            terms.append({'type': 'gp', 'c1': c1, 'c2': c2, 'a1': float(a1), 'a2': float(a2), 'p': 1, 'q': 1})
            feats.append((dx1 * dx2) * g)

    F = np.column_stack(feats) if feats else np.zeros((n, 0))
    return terms, F

def _term_to_string(term):
    if term['type'] == 'poly':
        p, q = term['p'], term['q']
        parts = []
        if p > 0:
            if p == 1:
                parts.append("x1")
            else:
                parts.append(f"x1**{p}")
        if q > 0:
            if q == 1:
                parts.append("x2")
            else:
                parts.append(f"x2**{q}")
        if not parts:
            return "1"
        return "*".join(parts)
    elif term['type'] == 'gp':
        c1 = _float_to_str(term['c1'])
        c2 = _float_to_str(term['c2'])
        a1 = _float_to_str(term['a1'])
        a2 = _float_to_str(term['a2'])
        p = term['p']
        q = term['q']
        dx1 = f"(x1 - {c1})"
        dx2 = f"(x2 - {c2})"
        exp_part = f"exp(-({a1}*{dx1}**2 + {a2}*{dx2}**2))"
        pref_parts = []
        if p == 1:
            pref_parts.append(dx1)
        elif p == 2:
            pref_parts.append(f"{dx1}**2")
        if q == 1:
            pref_parts.append(dx2)
        elif q == 2:
            pref_parts.append(f"{dx2}**2")
        if pref_parts:
            return "*".join(pref_parts + [exp_part])
        else:
            return exp_part
    else:
        return "1"

def _fit_least_squares(F, y):
    # Add intercept
    n = F.shape[0]
    intercept = np.ones((n, 1), dtype=float)
    A = np.hstack([intercept, F]) if F.size else intercept
    # Column scaling to improve conditioning (except intercept)
    col_scales = np.ones(A.shape[1], dtype=float)
    if A.shape[1] > 1:
        # compute L2 rms for each column (excluding intercept)
        norms = np.sqrt(np.mean(A[:, 1:] ** 2, axis=0) + 1e-18)
        norms[norms < 1e-9] = 1.0
        A[:, 1:] = A[:, 1:] / norms
        col_scales[1:] = norms
    # Solve least squares
    coef_scaled, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
    # Unscale
    coef = coef_scaled / col_scales
    return coef  # includes intercept at index 0

def _evaluate_selected_terms(X, selected_terms, coefs):
    # coefs includes intercept as first value, followed by one per selected term
    n = X.shape[0]
    y_pred = np.full(n, coefs[0], dtype=float)
    if not selected_terms:
        return y_pred
    x1 = X[:, 0]
    x2 = X[:, 1]
    for coef, term in zip(coefs[1:], selected_terms):
        if term['type'] == 'poly':
            p, q = term['p'], term['q']
            col = (x1 ** p) * (x2 ** q)
        elif term['type'] == 'gp':
            dx1 = x1 - term['c1']
            dx2 = x2 - term['c2']
            g = np.exp(- (term['a1'] * (dx1 ** 2) + term['a2'] * (dx2 ** 2)))
            mul = 1.0
            if term['p'] == 1:
                mul *= dx1
            elif term['p'] == 2:
                mul *= dx1 ** 2
            if term['q'] == 1:
                mul *= dx2
            elif term['q'] == 2:
                mul *= dx2 ** 2
            col = mul * g
        else:
            col = 0.0
        y_pred += coef * col
    return y_pred

def _build_expression_string(intercept, selected_terms, coefs):
    # coefs aligned with selected_terms, length = len(selected_terms)
    parts = []
    # Intercept
    if abs(intercept) > 1e-12:
        parts.append(_float_to_str(intercept))
    # Other terms
    for c, term in zip(coefs, selected_terms):
        term_str = _term_to_string(term)
        c_str = _float_to_str(c)
        parts.append(f"({c_str})*({term_str})")
    if not parts:
        return "0"
    return " + ".join(parts)

class Solution:
    def __init__(self, **kwargs):
        self.random_state = int(kwargs.get("random_state", 0))
        self.max_centers = int(kwargs.get("max_centers", 12))
        self.k_candidates = kwargs.get("k_candidates", [6, 8, 10, 12, 14])

    def solve(self, X: np.ndarray, y: np.ndarray) -> dict:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        n = X.shape[0]
        if X.shape[1] != 2:
            raise ValueError("X must have shape (n, 2)")
        # Build pool of candidate terms and features
        terms, F = _build_terms_and_features(X, max_centers=self.max_centers, seed=self.random_state)
        # Remove near-zero columns to avoid degeneracy
        if F.size > 0:
            col_norms = np.sqrt(np.mean(F ** 2, axis=0) + 1e-18)
            valid = col_norms > 1e-10
            F = F[:, valid]
            terms = [t for t, v in zip(terms, valid) if v]
        else:
            terms = []
        # Initial fit to all features to compute importances
        if len(terms) > 0:
            coef_all = _fit_least_squares(F, y)
            intercept_all = coef_all[0]
            weights_all = coef_all[1:]
            # Importance in scaled sense: use column RMS times |w|
            col_rms = np.sqrt(np.mean(F ** 2, axis=0) + 1e-18)
            importance = np.abs(weights_all) * col_rms
            order = np.argsort(-importance)
        else:
            intercept_all = float(np.mean(y))
            weights_all = np.array([])
            order = np.array([], dtype=int)

        # Choose number of terms via small search
        candidate_ks = [k for k in self.k_candidates if k <= len(order)]
        if not candidate_ks and len(order) > 0:
            candidate_ks = [min(6, len(order))]
        if not candidate_ks:
            # No terms; constant model
            expression = _float_to_str(intercept_all)
            predictions = np.full(n, intercept_all, dtype=float)
            return {
                "expression": expression,
                "predictions": predictions.tolist(),
                "details": {}
            }

        best_mse = None
        best_sel_idx = None
        best_coefs = None
        best_intercept = None

        for k in candidate_ks:
            sel = order[:k]
            F_sel = F[:, sel]
            coef_sel = _fit_least_squares(F_sel, y)
            intercept = coef_sel[0]
            weights = coef_sel[1:]
            y_pred = intercept + F_sel @ weights
            mse = float(np.mean((y_pred - y) ** 2))
            if (best_mse is None) or (mse < best_mse):
                best_mse = mse
                best_sel_idx = sel
                best_coefs = weights
                best_intercept = intercept

        # Build expression string from selected terms
        selected_terms = [terms[i] for i in best_sel_idx]
        expression = _build_expression_string(best_intercept, selected_terms, best_coefs)
        # Predictions
        predictions = _evaluate_selected_terms(X, selected_terms, np.concatenate([[best_intercept], best_coefs]))

        return {
            "expression": expression,
            "predictions": predictions.tolist(),
            "details": {}
        }