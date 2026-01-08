import pandas as pd
import numpy as np


class Solution:
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
        # Step 1: Apply column merges (if any)
        df_work = df.copy()

        if col_merge:
            for group in col_merge:
                if not group:
                    continue

                # Resolve column identifiers (names or integer positions)
                cols = []
                for c in group:
                    if isinstance(c, str):
                        if c in df_work.columns:
                            cols.append(c)
                    elif isinstance(c, int):
                        if 0 <= c < len(df_work.columns):
                            cols.append(df_work.columns[c])
                    # Ignore unsupported types silently

                # Remove duplicates while preserving order
                seen = set()
                cols = [c for c in cols if not (c in seen or seen.add(c))]

                if len(cols) <= 1:
                    continue

                # Merge the columns into the first column's position/name
                first_col = cols[0]
                try:
                    merged_series = df_work[cols].astype(str).agg("".join, axis=1)
                except Exception:
                    # Fallback row-wise concatenation if astype/agg fails
                    merged_series = df_work[cols].apply(
                        lambda row: "".join(str(v) for v in row.values), axis=1
                    )

                df_work[first_col] = merged_series
                # Drop the remaining columns in the group
                drop_cols = [c for c in cols[1:] if c in df_work.columns]
                if drop_cols:
                    df_work = df_work.drop(columns=drop_cols)

        # Step 2: Reorder columns based on heuristics
        N = len(df_work)
        if N == 0 or df_work.shape[1] <= 1:
            return df_work

        # Sampling for speed on very large datasets
        if N > early_stop:
            df_sample = df_work.iloc[:early_stop]
            N_sample = early_stop
        else:
            df_sample = df_work
            N_sample = N

        # Sample size used for estimating string lengths
        sample_size_for_len = min(N_sample, max(1000, row_stop * 1000))

        col_metrics = []

        for col in df_sample.columns:
            s = df_sample[col]

            # Value counts for probability and distinct count
            try:
                vc = s.value_counts(dropna=False)
                counts = vc.values.astype("float64", copy=False)
                unique_count = len(counts)
            except Exception:
                # Fallback if value_counts fails
                values = s.values
                unique_vals = pd.unique(values)
                unique_count = len(unique_vals)
                counts = np.array(
                    [np.sum(values == uv) for uv in unique_vals], dtype="float64"
                )

            if N_sample > 0:
                unique_ratio = unique_count / float(N_sample)
            else:
                unique_ratio = 0.0

            # Probability that two random rows share the same value in this column
            if N_sample > 0:
                p_same = float(np.sum(counts * counts)) / float(N_sample * N_sample)
            else:
                p_same = 0.0

            # Sequential run ratio: fraction of row-to-row changes
            if N_sample > 1:
                v = s.values
                prev = v[:-1]
                nxt = v[1:]
                try:
                    eq = np.asarray(prev == nxt)
                    same_count = int(eq.sum())
                    diff_count = (N_sample - 1) - same_count
                except Exception:
                    diff_count = 0
                    for a, b in zip(prev, nxt):
                        # Treat two NaNs as equal
                        if pd.isna(a) and pd.isna(b):
                            continue
                        if a != b:
                            diff_count += 1
                run_ratio = diff_count / float(N_sample - 1)
            else:
                run_ratio = 0.0

            # Average string length (approximate, using a sample)
            s_len_sample = s.iloc[:sample_size_for_len]
            try:
                avg_len = float(s_len_sample.astype(str).str.len().mean())
                if np.isnan(avg_len):
                    avg_len = 0.0
            except Exception:
                avg_len = 0.0

            # Score based on an independence-based heuristic:
            # g = (p_same * avg_len) / (1 - p_same)
            if p_same >= 0.999999999:
                g_score = float("inf")
            else:
                denom = 1.0 - p_same
                if denom <= 0.0:
                    denom = 1e-12
                g_score = (p_same * avg_len) / denom

            col_metrics.append((col, unique_ratio, run_ratio, avg_len, p_same, g_score))

        # Partition columns into low- and high-cardinality based on distinct_value_threshold
        low_cols = [m for m in col_metrics if m[1] <= distinct_value_threshold]
        high_cols = [m for m in col_metrics if m[1] > distinct_value_threshold]

        def sort_key_low(m):
            # m = (col, unique_ratio, run_ratio, avg_len, p_same, g_score)
            _, unique_ratio, run_ratio, avg_len, _, g_score = m
            if not np.isfinite(g_score):
                g_val = 1e30
            else:
                g_val = g_score
            # Primary: descending g_score, then ascending run_ratio, ascending unique_ratio,
            # then descending avg_len
            return (-g_val, run_ratio, unique_ratio, -avg_len)

        def sort_key_high(m):
            _, unique_ratio, run_ratio, avg_len, _, g_score = m
            if not np.isfinite(g_score):
                g_val = 1e30
            else:
                g_val = g_score
            return (-g_val, run_ratio, unique_ratio, -avg_len)

        low_sorted = sorted(low_cols, key=sort_key_low)
        high_sorted = sorted(high_cols, key=sort_key_high)

        new_order = [m[0] for m in low_sorted] + [m[0] for m in high_sorted]

        # Ensure all columns are present in the final order
        seen_cols = set(new_order)
        for col in df_work.columns:
            if col not in seen_cols:
                new_order.append(col)
                seen_cols.add(col)

        df_reordered = df_work[new_order]
        return df_reordered