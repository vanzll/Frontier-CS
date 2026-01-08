import pandas as pd
import math
from typing import List, Tuple, Dict, Set


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
        # Handle trivial cases
        if df is None or df.shape[1] <= 1 or df.shape[0] == 0:
            return df

        # Step 1: Apply column merges if specified
        df_work = df.copy()
        if col_merge:
            df_work = self._apply_col_merges(df_work, col_merge)

        # Prepare string arrays for each column
        colnames = list(df_work.columns)
        N = len(df_work)
        if N == 0:
            return df_work

        # Convert entire DataFrame to string arrays per column efficiently
        col_vals = [self._series_to_str_list(df_work[c]) for c in colnames]

        # Precompute column metrics for tie-breaking
        col_metrics = self._compute_column_metrics(col_vals)

        # Determine number of rows to consider in heuristic scoring
        Nlimit = min(N, max(1, int(early_stop)))

        # Greedy selection of columns to maximize incremental LCP contribution
        remaining = list(range(len(colnames)))
        selected = []
        keys = [tuple()] * Nlimit  # keys over selected columns, per row (sampled)

        while remaining:
            best_cand = None
            best_score = -1
            best_eq_full_len = -1  # tie-breaker: sum of full-match lengths
            for c in remaining:
                score, eq_full_len = self._candidate_incremental_score(
                    values=[col_vals[c][i] for i in range(Nlimit)],
                    keys=keys,
                )
                if score > best_score:
                    best_score = score
                    best_eq_full_len = eq_full_len
                    best_cand = c
                elif score == best_score:
                    # Tie-breaker: prefer columns that produce more full matches
                    if eq_full_len > best_eq_full_len:
                        best_eq_full_len = eq_full_len
                        best_cand = c
                    elif eq_full_len == best_eq_full_len:
                        # Further tie-breaker using precomputed global metrics
                        # Prefer lower distinct ratio, higher P_eq, then higher avg length
                        dc = col_metrics[c]
                        db = col_metrics[best_cand] if best_cand is not None else (-1, -1, -1)
                        # dc/db = (uniq_ratio, P_eq, avg_len)
                        if (
                            dc[0] < db[0]
                            or (dc[0] == db[0] and dc[1] > db[1])
                            or (dc[0] == db[0] and dc[1] == db[1] and dc[2] > db[2])
                        ):
                            best_cand = c
            selected.append(best_cand)
            remaining.remove(best_cand)
            # Update keys by appending chosen column values
            vals_best = [col_vals[best_cand][i] for i in range(Nlimit)]
            keys = [keys[i] + (vals_best[i],) for i in range(Nlimit)]

        # Reorder DataFrame columns according to selected indices
        ordered_cols = [colnames[i] for i in selected]
        df_out = df_work.loc[:, ordered_cols]
        return df_out

    def _series_to_str_list(self, s: pd.Series) -> List[str]:
        arr = s.to_numpy()
        res = []
        append = res.append
        for x in arr:
            if pd.isna(x):
                append("")
            else:
                append(str(x))
        return res

    def _apply_col_merges(self, df: pd.DataFrame, col_merge: list) -> pd.DataFrame:
        df_work = df.copy()
        existing = set(df_work.columns)
        for idx, group in enumerate(col_merge):
            if not group:
                continue
            real_group = [c for c in group if c in df_work.columns]
            if not real_group:
                continue
            # Build merged series
            merged = None
            for c in real_group:
                s = df_work[c].astype('string').fillna('')
                if merged is None:
                    merged = s
                else:
                    merged = merged + s
            # Determine a unique new column name
            base_name = f"__MERGED__{idx}"
            new_name = base_name
            k = 1
            while new_name in existing:
                new_name = f"{base_name}_{k}"
                k += 1
            # Drop old columns and insert merged column
            df_work = df_work.drop(columns=real_group)
            df_work[new_name] = merged.astype(object)
            existing = set(df_work.columns)
        return df_work

    def _compute_column_metrics(self, col_vals: List[List[str]]) -> Dict[int, Tuple[float, float, float]]:
        # Returns dict: idx -> (uniq_ratio, P_eq, avg_length)
        metrics = {}
        N = len(col_vals[0]) if col_vals else 0
        if N == 0:
            return {i: (1.0, 0.0, 0.0) for i in range(len(col_vals))}
        for idx, vals in enumerate(col_vals):
            lengths = [len(v) for v in vals]
            avg_len = sum(lengths) / N if N else 0.0
            counts = {}
            for v in vals:
                counts[v] = counts.get(v, 0) + 1
            uniq_ratio = len(counts) / N
            # Probability that two random samples are equal: sum (p_i^2)
            P_eq = 0.0
            invN = 1.0 / N
            for c in counts.values():
                p = c * invN
                P_eq += p * p
            metrics[idx] = (uniq_ratio, P_eq, avg_len)
        return metrics

    def _candidate_incremental_score(
        self,
        values: List[str],
        keys: List[Tuple[str, ...]],
    ) -> Tuple[int, int]:
        # groups: key -> (prefix_set, value_set)
        prefix_sets: Dict[Tuple[str, ...], Set[str]] = {}
        value_sets: Dict[Tuple[str, ...], Set[str]] = {}
        total_lcp = 0
        fullmatch_len_sum = 0
        for i, s in enumerate(values):
            k = keys[i]
            ps = prefix_sets.get(k)
            lcp_len = 0
            if ps:
                # Compute longest k such that s[:k] in ps
                # Fast-loop scanning prefixes
                # Break at first missing prefix
                # Heuristic: check first char quickly
                # But generic loop suffices
                for j in range(1, len(s) + 1):
                    if s[:j] in ps:
                        lcp_len = j
                    else:
                        break
                vs = value_sets.get(k)
                if vs and s in vs:
                    fullmatch_len_sum += len(s)
            total_lcp += lcp_len
            # Update structures
            ps = prefix_sets.setdefault(k, set())
            # Add all prefixes of s
            # Avoid overhead when s is empty string
            if s:
                for j in range(1, len(s) + 1):
                    ps.add(s[:j])
            vs = value_sets.setdefault(k, set())
            vs.add(s)
        return total_lcp, fullmatch_len_sum