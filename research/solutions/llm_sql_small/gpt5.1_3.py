import pandas as pd
import numpy as np


class Solution:
    def __init__(self):
        self._max_lcp_check = 64

    def _lcp_len(self, a, b):
        if not isinstance(a, str):
            a = str(a)
        if not isinstance(b, str):
            b = str(b)
        max_check = self._max_lcp_check
        la = len(a)
        lb = len(b)
        n = la if la < lb else lb
        if n > max_check:
            n = max_check
        i = 0
        while i < n and a[i] == b[i]:
            i += 1
        return i

    def _avg_best_lcp_window(self, arr, row_stop):
        n = len(arr)
        if n <= 1:
            return 0.0
        if row_stop is None or row_stop <= 0:
            row_stop = 1
        row_stop = int(row_stop)
        if row_stop < 1:
            row_stop = 1
        if row_stop > 16:
            row_stop = 16
        total = 0
        for i in range(1, n):
            best = 0
            start = i - row_stop
            if start < 0:
                start = 0
            ai = arr[i]
            for j in range(start, i):
                l = self._lcp_len(ai, arr[j])
                if l > best:
                    best = l
            total += best
        return total / (n - 1)

    def _apply_col_merge(self, df, col_merge):
        if not col_merge:
            return df.copy()
        df_work = df.copy()
        drop_cols = set()
        for group in col_merge:
            if not group:
                continue
            cols = [c for c in group if c in df_work.columns and c not in drop_cols]
            if len(cols) <= 1:
                continue
            base = cols[0]
            merged = df_work[cols].astype(str).agg(''.join, axis=1)
            df_work[base] = merged
            for c in cols[1:]:
                drop_cols.add(c)
        if drop_cols:
            exist = [c for c in df_work.columns if c in drop_cols]
            if exist:
                df_work = df_work.drop(columns=exist)
        return df_work

    def _compute_column_metrics(self, df, row_stop, distinct_value_threshold):
        n_rows = len(df)
        cols = list(df.columns)
        metrics = {}
        if n_rows == 0:
            for name in cols:
                metrics[name] = {
                    'p_eff': 0.0,
                    'mean_len': 0.0,
                }
            return metrics
        for name in cols:
            ser = df[name]
            ser_str = ser.astype(str)
            arr = ser_str.to_numpy(copy=False)

            try:
                lengths = np.fromiter((len(x) for x in arr), dtype=np.float64, count=n_rows)
            except TypeError:
                lengths = ser_str.str.len().to_numpy()

            if lengths.size == 0:
                mean_len = 0.0
            else:
                mean_len = float(lengths.mean())

            vc = ser_str.value_counts(dropna=False)
            freqs = vc.to_numpy(dtype=np.float64)
            if freqs.size == 0:
                p_same = 0.0
                distinct_ratio = 0.0
            else:
                probs = freqs / float(n_rows)
                p_same = float(np.sum(probs * probs))
                distinct_ratio = float(len(freqs) / float(n_rows))

            if mean_len > 0 and n_rows > 1:
                avg_lcp = self._avg_best_lcp_window(arr, row_stop)
                local_sim = avg_lcp / mean_len
                if local_sim > 1.0:
                    local_sim = 1.0
                elif local_sim < 0.0:
                    local_sim = 0.0
            else:
                local_sim = 0.0

            if distinct_value_threshold is None:
                distinct_value_threshold = 0.7
            try:
                thr = float(distinct_value_threshold)
            except Exception:
                thr = 0.7
            if thr < 0:
                thr = 0.0
            if thr > 1:
                thr = 1.0

            if distinct_ratio <= thr or thr >= 1.0:
                h = 0.0
            else:
                denom = 1.0 - thr
                if denom <= 0:
                    h = 0.0
                else:
                    h = (distinct_ratio - thr) / denom
                    if h < 0:
                        h = 0.0
                    elif h > 1.0:
                        h = 1.0

            p_eff = (1.0 - h) * p_same + h * local_sim
            if p_eff < 0.0:
                p_eff = 0.0
            elif p_eff > 1.0:
                p_eff = 1.0

            metrics[name] = {
                'p_eff': p_eff,
                'mean_len': mean_len,
            }
        return metrics

    def _find_best_order_dp(self, columns, metrics):
        m = len(columns)
        if m <= 1:
            return list(columns)
        P = np.empty(m, dtype=np.float64)
        L = np.empty(m, dtype=np.float64)
        for idx, col in enumerate(columns):
            info = metrics[col]
            P[idx] = info['p_eff']
            L[idx] = info['mean_len']
        w = P * L
        n_masks = 1 << m
        prod = [1.0] * n_masks
        for mask in range(1, n_masks):
            lsb = mask & -mask
            i = (lsb.bit_length() - 1)
            prev = mask ^ lsb
            prod[mask] = prod[prev] * P[i]
        bestF = [0.0] * n_masks
        choice = [-1] * n_masks
        bestF[0] = 0.0
        for mask in range(1, n_masks):
            best_val = -1e300
            best_idx = -1
            sub = mask
            while sub:
                lsb = sub & -sub
                i = (lsb.bit_length() - 1)
                prev = mask ^ lsb
                cand = bestF[prev] + prod[prev] * w[i]
                if cand > best_val:
                    best_val = cand
                    best_idx = i
                sub ^= lsb
            bestF[mask] = best_val
            choice[mask] = best_idx
        mask = n_masks - 1
        order_indices_reversed = []
        while mask:
            i = choice[mask]
            if i < 0:
                lsb = mask & -mask
                i = (lsb.bit_length() - 1)
            order_indices_reversed.append(i)
            mask ^= (1 << i)
        order_indices = list(reversed(order_indices_reversed))
        ordered_cols = [columns[i] for i in order_indices]
        return ordered_cols

    def solve(
        self,
        df: pd.DataFrame,
        early_stop: int = 100000,
        row_stop: int = 4,
        col_stop: int = 2,
        col_merge: list = None,
        one_way_dep: list = None,
        distinct_value_threshold: float = 0.7,
        parallel: bool = True,
    ) -> pd.DataFrame:
        if col_merge:
            df_work = self._apply_col_merge(df, col_merge)
        else:
            df_work = df.copy()

        if df_work.shape[1] <= 1:
            return df_work

        metrics = self._compute_column_metrics(
            df_work,
            row_stop=row_stop,
            distinct_value_threshold=distinct_value_threshold,
        )
        cols = list(df_work.columns)
        best_order = self._find_best_order_dp(cols, metrics)
        best_order = [c for c in best_order if c in df_work.columns]
        return df_work[best_order]