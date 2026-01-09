import numpy as np

class Solution:
    def __init__(self, **kwargs):
        pass

    def solve(self, X: np.ndarray, y: np.ndarray) -> dict:
        if X.size == 0 or y.size == 0:
            return {
                "expression": "0",
                "predictions": [],
                "details": {}
            }

        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        x1 = X[:, 0]
        x2 = X[:, 1]

        ones = np.ones_like(x1, dtype=float)

        features = []
        features.append(("1", ones))  # index 0: constant
        features.append(("x1", x1))
        features.append(("x2", x2))
        features.append(("sin(x1)", np.sin(x1)))
        features.append(("cos(x1)", np.cos(x1)))
        features.append(("sin(x2)", np.sin(x2)))
        features.append(("cos(x2)", np.cos(x2)))
        features.append(("sin(x1)*sin(x2)", np.sin(x1) * np.sin(x2)))
        features.append(("sin(x1)*cos(x2)", np.sin(x1) * np.cos(x2)))
        features.append(("cos(x1)*sin(x2)", np.cos(x1) * np.sin(x2)))
        features.append(("cos(x1)*cos(x2)", np.cos(x1) * np.cos(x2)))
        features.append(("sin(x1 + x2)", np.sin(x1 + x2)))
        features.append(("sin(x1 - x2)", np.sin(x1 - x2)))
        features.append(("cos(x1 + x2)", np.cos(x1 + x2)))
        features.append(("cos(x1 - x2)", np.cos(x1 - x2)))
        features.append(("sin(2*x1)", np.sin(2.0 * x1)))
        features.append(("cos(2*x1)", np.cos(2.0 * x1)))
        features.append(("sin(2*x2)", np.sin(2.0 * x2)))
        features.append(("cos(2*x2)", np.cos(2.0 * x2)))
        features.append(("x1*x2", x1 * x2))

        F = np.column_stack([f[1] for f in features])
        n, p = F.shape

        def orthogonal_matching_pursuit(F_mat, y_vec, max_terms_nonconst=8):
            n_local, p_local = F_mat.shape
            include_constant = True

            if include_constant:
                active = [0]
            else:
                active = []

            if active:
                F_active = F_mat[:, active]
                w_active, *_ = np.linalg.lstsq(F_active, y_vec, rcond=None)
                residual = y_vec - F_active @ w_active
            else:
                w_active = np.zeros(0, dtype=float)
                residual = y_vec.copy()

            mse = float(np.mean(residual ** 2))
            saved = [(list(active), w_active.copy(), mse)]

            norms = np.linalg.norm(F_mat, axis=0)
            max_terms_nonconst = min(max_terms_nonconst, max(0, p_local - (1 if include_constant else 0)))

            while True:
                if (len(active) - (1 if include_constant else 0)) >= max_terms_nonconst:
                    break

                best_j = None
                best_score = 0.0
                r = residual

                for j in range(p_local):
                    if j in active:
                        continue
                    nj = norms[j]
                    if not np.isfinite(nj) or nj == 0.0:
                        continue
                    corr = abs(np.dot(F_mat[:, j], r) / nj)
                    if corr > best_score:
                        best_score = corr
                        best_j = j

                if best_j is None or best_score <= 0.0:
                    break

                active.append(best_j)
                F_active = F_mat[:, active]
                w_active, *_ = np.linalg.lstsq(F_active, y_vec, rcond=None)
                residual = y_vec - F_active @ w_active
                mse = float(np.mean(residual ** 2))
                saved.append((list(active), w_active.copy(), mse))

            mses = [m for (_, _, m) in saved]
            min_mse = min(mses)
            var_y = float(np.var(y_vec))
            scale = max(1.0, var_y)
            floor = 1e-10 * scale

            if min_mse <= floor:
                allowed_mse = min_mse + floor
            else:
                allowed_mse = min_mse * 1.01

            candidates = []
            for (idxs, w_vec, m) in saved:
                num_nonconst = len(idxs) - (1 if include_constant else 0)
                if m <= allowed_mse:
                    candidates.append((num_nonconst, m, idxs, w_vec))

            if not candidates:
                for (idxs, w_vec, m) in saved:
                    num_nonconst = len(idxs) - (1 if include_constant else 0)
                    candidates.append((num_nonconst, m, idxs, w_vec))

            candidates.sort(key=lambda t: (t[0], t[1]))
            _, best_mse_local, best_idxs, best_w = candidates[0]
            return best_idxs, best_w, best_mse_local

        best_indices, best_w, _ = orthogonal_matching_pursuit(F, y, max_terms_nonconst=8)

        feat_indices = []
        coeffs = []
        for pos, idx in enumerate(best_indices):
            c = float(best_w[pos])
            c_rounded = float(np.round(c, 8))
            if abs(c_rounded) < 1e-8:
                continue
            feat_indices.append(idx)
            coeffs.append(c_rounded)

        if not coeffs:
            expression = "0"
            predictions = np.zeros_like(y, dtype=float)
            return {
                "expression": expression,
                "predictions": predictions.tolist(),
                "details": {}
            }

        if 0 in feat_indices:
            pos0 = feat_indices.index(0)
            if pos0 != 0:
                feat_indices.insert(0, feat_indices.pop(pos0))
                coeffs.insert(0, coeffs.pop(pos0))

        expression = ""
        first = True
        for idx, c in zip(feat_indices, coeffs):
            is_constant = (idx == 0)
            if c < 0:
                sign = "-"
                mag = -c
            else:
                sign = "+"
                mag = c

            if is_constant:
                term_body = repr(mag)
            else:
                term_str = features[idx][0]
                if abs(mag - 1.0) < 1e-8:
                    term_body = term_str
                else:
                    term_body = f"{repr(mag)}*{term_str}"

            if first:
                if sign == "-":
                    expression += "-" + term_body
                else:
                    expression += term_body
                first = False
            else:
                if sign == "-":
                    expression += " - " + term_body
                else:
                    expression += " + " + term_body

        if expression == "":
            expression = "0"

        F_selected = F[:, feat_indices]
        coef_vec = np.array(coeffs, dtype=float)
        predictions = F_selected @ coef_vec

        return {
            "expression": expression,
            "predictions": predictions.tolist(),
            "details": {}
        }