import pandas as pd
import numpy as np


class Solution:
    def _apply_col_merge(self, df: pd.DataFrame, col_merge, orig_cols):
        if not col_merge:
            return
        for group in col_merge:
            if not group:
                continue
            # Resolve column names from group (can be names or indices)
            names = []
            for g in group:
                if isinstance(g, int):
                    if 0 <= g < len(orig_cols):
                        nm = orig_cols[g]
                        if nm in df.columns:
                            names.append(nm)
                else:
                    if g in df.columns:
                        names.append(g)
            # Deduplicate while preserving order
            seen = set()
            unique_names = []
            for nm in names:
                if nm not in seen:
                    unique_names.append(nm)
                    seen.add(nm)
            names = unique_names
            if len(names) <= 1:
                continue

            base_name = names[0]
            base_arr = df[base_name].astype(str).to_numpy()
            merged = base_arr
            for nm in names[1:]:
                arr = df[nm].astype(str).to_numpy()
                merged = np.char.add(merged, arr)
            df[base_name] = merged

            drop_cols = [nm for nm in names[1:] if nm in df.columns]
            if drop_cols:
                df.drop(columns=drop_cols, inplace=True)

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
        """
        Reorder columns in the DataFrame to maximize prefix hit rate.
        """
        if df is None or df.shape[0] == 0 or df.shape[1] <= 1:
            return df

        # Work on a copy to avoid mutating input
        df_work = df.copy()
        orig_cols = list(df_work.columns)

        # Apply column merges if provided
        if col_merge:
            self._apply_col_merge(df_work, col_merge, orig_cols)

        cols = list(df_work.columns)
        M = len(cols)
        if M <= 1:
            return df_work

        N = len(df_work)

        # Precompute string representations, distinct ratios, and average lengths
        str_cols = []
        uniq_ratios = []
        avg_lens = []

        for col in cols:
            s = df_work[col].astype(str)
            arr = s.to_numpy(dtype=object, copy=False)
            str_cols.append(arr)

            try:
                uniq = s.nunique(dropna=False)
            except TypeError:
                uniq = len(pd.unique(s))

            ratio = float(uniq) / float(N) if N > 0 else 0.0
            if ratio > 1.0:
                ratio = 1.0
            uniq_ratios.append(ratio)

            total_len = 0
            for v in arr:
                total_len += len(v)
            avg_lens.append(total_len / N if N > 0 else 0.0)

        # Build initial order based on distinctness and length
        indices = list(range(M))
        low_indices = [i for i in indices if uniq_ratios[i] <= distinct_value_threshold]
        high_indices = [i for i in indices if i not in low_indices]

        low_indices.sort(key=lambda i: (uniq_ratios[i], -avg_lens[i]))
        if high_indices:
            high_indices.sort(key=lambda i: (avg_lens[i], uniq_ratios[i]))
        initial_order = low_indices + high_indices

        # Sampling configuration for approximate scoring
        row_len_avg = sum(avg_lens)
        max_iters = col_stop if isinstance(col_stop, int) and col_stop > 0 else 1
        perm_est = 1 + max_iters * (M * (M - 1) // 2)

        # Baseline sample_n from row_stop
        if isinstance(row_stop, int) and row_stop > 0:
            sample_n = min(N, max(1000, row_stop * 1000))
        else:
            sample_n = min(N, 4000)

        # Apply early_stop as an upper bound on sampled rows
        if isinstance(early_stop, int) and early_stop > 0:
            sample_n = min(sample_n, early_stop)

        # Adjust sample size based on estimated character budget
        if row_len_avg > 0 and N > 0 and perm_est > 0:
            target_total_chars = 20_000_000  # heuristic budget
            max_sample_by_chars = int(target_total_chars / (perm_est * row_len_avg))
            if max_sample_by_chars < 100:
                max_sample_by_chars = 100
            if max_sample_by_chars < 1:
                max_sample_by_chars = 1
            if sample_n > max_sample_by_chars:
                sample_n = max_sample_by_chars

        if sample_n < 1:
            sample_n = min(N, 1)
        if sample_n > N:
            sample_n = N

        if sample_n >= N:
            sample_rows = np.arange(N, dtype=int)
        else:
            rng = np.random.RandomState(0)
            sample_rows = np.sort(rng.choice(N, size=sample_n, replace=False))

        str_cols_local = str_cols
        rows_local = sample_rows

        def approximate_score(order_idx):
            root = {}
            total_lcp = 0
            col_data = str_cols_local
            rows = rows_local
            processed = 0
            for r in rows:
                if isinstance(early_stop, int) and early_stop > 0 and processed >= early_stop:
                    break
                processed += 1

                s = ''.join(col_data[c][r] for c in order_idx)
                node = root
                lcp = 0
                matched = True
                for ch in s:
                    child = node.get(ch)
                    if child is not None:
                        node = child
                        if matched:
                            lcp += 1
                    else:
                        child = {}
                        node[ch] = child
                        node = child
                        matched = False
                total_lcp += lcp
            return total_lcp

        # Hill-climbing local search
        best_order = initial_order
        best_score = approximate_score(best_order)

        if M >= 2:
            for _ in range(max_iters):
                improved = False
                base_order = best_order
                for i in range(M):
                    for j in range(i + 1, M):
                        cand = list(base_order)
                        cand[i], cand[j] = cand[j], cand[i]
                        score = approximate_score(cand)
                        if score > best_score:
                            best_score = score
                            best_order = cand
                            improved = True
                if not improved:
                    break

        ordered_cols = [cols[i] for i in best_order]
        df_reordered = df_work[ordered_cols]
        return df_reordered