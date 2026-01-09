import numpy as np

class Solution:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def solve(self, X: np.ndarray, y: np.ndarray) -> dict:
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).reshape(-1)
        n = X.shape[0]
        x1 = X[:, 0]
        x2 = X[:, 1]

        r2 = x1 * x1 + x2 * x2
        r2_2 = r2 * r2
        r2_3 = r2_2 * r2
        r = np.sqrt(r2)

        def lstsq_fit(cols):
            A = np.column_stack(cols)
            coeffs, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
            pred = A @ coeffs
            err = y - pred
            mse = float(np.mean(err * err))
            return coeffs, pred, mse, A

        # Baseline linear regression for fallback decision
        A_lin = np.column_stack([x1, x2, np.ones_like(x1)])
        c_lin, _, _, _ = np.linalg.lstsq(A_lin, y, rcond=None)
        pred_lin = A_lin @ c_lin
        err_lin = y - pred_lin
        mse_lin = float(np.mean(err_lin * err_lin))

        best = {
            "mse": float("inf"),
            "criterion": float("inf"),
            "coeffs": None,
            "pred": None,
            "basis_strs": None,
            "A": None,
            "keep_idx": None,
            "radial_kind": None,
            "omega": None,
            "model": None,
        }

        def update_best(mse, ncols, coeffs, pred, basis_strs, A, radial_kind, omega, model):
            criterion = mse * (1.0 + 0.002 * ncols)
            if (criterion < best["criterion"] * (1.0 - 1e-12)) or (
                abs(criterion - best["criterion"]) <= best["criterion"] * 1e-4 and ncols < len(best["basis_strs"]) if best["basis_strs"] is not None else True
            ):
                best.update(
                    mse=mse,
                    criterion=criterion,
                    coeffs=coeffs,
                    pred=pred,
                    basis_strs=basis_strs,
                    A=A,
                    keep_idx=list(range(ncols)),
                    radial_kind=radial_kind,
                    omega=omega,
                    model=model,
                )

        # Polynomial-only model on r2
        cols_poly = [np.ones_like(r2), r2, r2_2, r2_3]
        basis_poly = ["1", "(x1**2 + x2**2)", "(x1**2 + x2**2)**2", "(x1**2 + x2**2)**3"]
        coeffs, pred, mse, A = lstsq_fit(cols_poly)
        update_best(mse, len(cols_poly), coeffs, pred, basis_poly, A, "r2", None, "poly_r2")

        # Candidate omegas based on range
        def gen_omegas(range_val, max_cycles=60):
            range_val = float(range_val)
            if not np.isfinite(range_val) or range_val < 1e-12:
                return np.array([1.0], dtype=np.float64)
            cycles = np.linspace(1.0, float(max_cycles), int(max_cycles))
            omegas = 2.0 * np.pi * cycles / range_val
            extras = 2.0 * np.pi * np.array([0.5, 0.75, 1.25, 1.5, 2.0, 3.0, 5.0], dtype=np.float64) / range_val
            omegas = np.unique(np.concatenate([omegas, extras]))
            omegas = omegas[(omegas > 0) & np.isfinite(omegas)]
            return omegas

        r2_range = float(np.max(r2) - np.min(r2)) if n > 0 else 0.0
        r_range = float(np.max(r) - np.min(r)) if n > 0 else 0.0

        omegas_r2 = gen_omegas(r2_range, max_cycles=60)
        omegas_r = gen_omegas(r_range, max_cycles=70)

        # Model templates (names, columns builder, basis strings builder)
        def fit_trig_family(radial_kind, omega):
            if radial_kind == "r2":
                radial_arr = r2
                radial_expr = "(x1**2 + x2**2)"
            else:
                radial_arr = r
                radial_expr = "((x1**2 + x2**2)**0.5)"

            arg = omega * radial_arr
            s = np.sin(arg)
            c = np.cos(arg)

            omega_str = f"{float(omega):.12g}"
            arg_expr = f"({omega_str})*{radial_expr}"

            sin_str = f"sin({arg_expr})"
            cos_str = f"cos({arg_expr})"

            # Types: sc, linamp, quadamp
            models = []

            # sc
            cols = [s, c, np.ones_like(s)]
            basis = [sin_str, cos_str, "1"]
            models.append(("sc", cols, basis))

            # linamp
            cols = [s, r2 * s, c, r2 * c, np.ones_like(s), r2]
            basis = [
                sin_str,
                f"({basis_poly[1]})*({sin_str})",
                cos_str,
                f"({basis_poly[1]})*({cos_str})",
                "1",
                basis_poly[1],
            ]
            models.append(("linamp", cols, basis))

            # quadamp
            cols = [s, r2 * s, r2_2 * s, c, r2 * c, r2_2 * c, np.ones_like(s), r2, r2_2]
            basis = [
                sin_str,
                f"({basis_poly[1]})*({sin_str})",
                f"({basis_poly[2]})*({sin_str})",
                cos_str,
                f"({basis_poly[1]})*({cos_str})",
                f"({basis_poly[2]})*({cos_str})",
                "1",
                basis_poly[1],
                basis_poly[2],
            ]
            models.append(("quadamp", cols, basis))

            for model_name, cols, basis in models:
                coeffs, pred, mse, A = lstsq_fit(cols)
                update_best(mse, len(cols), coeffs, pred, basis, A, radial_kind, omega, f"{model_name}_{radial_kind}")

        # Scan omegas; stop early if very good fit
        target_good = mse_lin * 0.02  # very strong improvement
        max_scan = int(self.kwargs.get("max_scan", 220))

        scanned = 0
        for omega in omegas_r2:
            fit_trig_family("r2", float(omega))
            scanned += 1
            if best["mse"] <= target_good or scanned >= max_scan:
                break

        if scanned < max_scan and best["mse"] > target_good:
            for omega in omegas_r:
                fit_trig_family("r", float(omega))
                scanned += 1
                if best["mse"] <= target_good or scanned >= max_scan:
                    break

        # Greedy term elimination for simplicity, with small MSE tolerance
        coeffs = best["coeffs"]
        A = best["A"]
        basis_strs = best["basis_strs"]
        if coeffs is None or A is None or basis_strs is None:
            expression = f"{float(c_lin[0]):.12g}*x1 + {float(c_lin[1]):.12g}*x2 + {float(c_lin[2]):.12g}"
            return {"expression": expression, "predictions": pred_lin.tolist(), "details": {"mse": mse_lin, "model": "linear"}}

        idx = list(range(A.shape[1]))
        constant_indices = [i for i, b in enumerate(basis_strs) if b == "1"]

        def refit_for_idx(idxs):
            A_sub = A[:, idxs]
            c_sub, _, _, _ = np.linalg.lstsq(A_sub, y, rcond=None)
            pred_sub = A_sub @ c_sub
            err = y - pred_sub
            mse_sub = float(np.mean(err * err))
            return c_sub, pred_sub, mse_sub

        c_cur, pred_cur, mse_cur = refit_for_idx(idx)
        tol_abs = max(1e-10, mse_cur * 0.002)

        # Try removing terms while keeping MSE within tolerance
        for _ in range(len(idx) - 1):
            if len(idx) <= 1:
                break
            best_removal = None
            best_removal_mse = None
            best_removal_c = None
            best_removal_pred = None

            for j in idx:
                # keep at least one constant term if present
                if j in constant_indices and len(constant_indices) > 0:
                    remaining_consts = [k for k in constant_indices if k in idx and k != j]
                    if len(remaining_consts) == 0:
                        continue
                idx2 = [k for k in idx if k != j]
                if len(idx2) == 0:
                    continue
                c2, p2, m2 = refit_for_idx(idx2)
                if m2 <= mse_cur + tol_abs:
                    if best_removal is None or m2 < best_removal_mse * (1.0 - 1e-12) or (abs(m2 - mse_cur) < abs(best_removal_mse - mse_cur) and len(idx2) < len(idx)):
                        best_removal = j
                        best_removal_mse = m2
                        best_removal_c = c2
                        best_removal_pred = p2

            if best_removal is None:
                break

            idx = [k for k in idx if k != best_removal]
            c_cur, pred_cur, mse_cur = best_removal_c, best_removal_pred, best_removal_mse
            tol_abs = max(1e-10, mse_cur * 0.002)

        # Build expression
        def fmt_num(v):
            v = float(v)
            if not np.isfinite(v):
                return "0.0"
            s = f"{v:.12g}"
            if s == "-0":
                s = "0"
            return s

        kept_basis = [basis_strs[i] for i in idx]
        kept_coeffs = c_cur

        # Prune tiny coefficients
        max_abs = float(np.max(np.abs(kept_coeffs))) if kept_coeffs.size else 0.0
        tiny = max(1e-12, max_abs * 1e-10)
        pruned_terms = []
        for coef, bas in zip(kept_coeffs, kept_basis):
            if abs(float(coef)) <= tiny:
                continue
            pruned_terms.append((float(coef), bas))

        if not pruned_terms:
            expression = "0"
            predictions = np.zeros_like(y)
            mse_final = float(np.mean((y - predictions) ** 2))
            details = {"mse": mse_final, "model": "zero"}
            return {"expression": expression, "predictions": predictions.tolist(), "details": details}

        expr_parts = []
        for k, (coef, bas) in enumerate(pruned_terms):
            coef_str = fmt_num(abs(coef))
            if bas == "1":
                term = coef_str
            else:
                term = f"{coef_str}*({bas})"
            if k == 0:
                if coef < 0:
                    expr_parts.append(f"-({term})")
                else:
                    expr_parts.append(term)
            else:
                if coef < 0:
                    expr_parts.append(f"-({term})")
                else:
                    expr_parts.append(f"+({term})")
        expression = "".join(expr_parts)

        # Recompute predictions from pruned model using available A columns if possible, else use pred_cur
        # (pruning might have removed some terms; compute from y via design matrix if needed)
        if len(pruned_terms) == len(kept_basis):
            predictions = pred_cur
            mse_final = mse_cur
        else:
            # Build matrix for pruned terms from A columns where possible
            pruned_indices = []
            for coef, bas in pruned_terms:
                # find first occurrence in kept_basis with matching bas
                for ii, b in enumerate(kept_basis):
                    if b == bas:
                        pruned_indices.append(idx[ii])
                        break
            A_sub = A[:, pruned_indices]
            c_sub, _, _, _ = np.linalg.lstsq(A_sub, y, rcond=None)
            predictions = A_sub @ c_sub
            err = y - predictions
            mse_final = float(np.mean(err * err))

        details = {
            "mse": float(mse_final),
            "baseline_mse": float(mse_lin),
            "model": best["model"],
            "radial": best["radial_kind"],
            "omega": None if best["omega"] is None else float(best["omega"]),
            "n_terms": int(len(pruned_terms)),
        }

        return {
            "expression": expression,
            "predictions": predictions.tolist(),
            "details": details,
        }