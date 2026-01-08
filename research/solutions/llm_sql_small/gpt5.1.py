import pandas as pd
import itertools


class Solution:
    def _apply_merges(self, df: pd.DataFrame, col_merge: list) -> pd.DataFrame:
        if not col_merge:
            return df

        new_df = df.copy()

        for group in col_merge:
            if not isinstance(group, (list, tuple)):
                continue

            cols_present = []
            for g in group:
                if isinstance(g, int):
                    if 0 <= g < len(new_df.columns):
                        col_name = new_df.columns[g]
                        if col_name not in cols_present:
                            cols_present.append(col_name)
                else:
                    if g in new_df.columns and g not in cols_present:
                        cols_present.append(g)

            if len(cols_present) <= 1:
                continue

            base_name = "||".join(str(c) for c in cols_present)
            new_name = base_name
            suffix = 1
            while new_name in new_df.columns:
                new_name = f"{base_name}__{suffix}"
                suffix += 1

            arr = new_df[cols_present].astype(str).values
            merged = ["".join(row) for row in arr]
            new_df[new_name] = merged
            new_df = new_df.drop(columns=cols_present)

        return new_df

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
        work_df = df.copy()

        if col_merge:
            work_df = self._apply_merges(work_df, col_merge)

        n_rows, n_cols = work_df.shape
        if n_cols <= 1 or n_rows == 0:
            return work_df

        str_df = work_df.astype(str)
        cols = list(work_df.columns)
        N = float(n_rows)

        p_match = {}
        avg_len = {}

        for col in cols:
            s = str_df[col]

            lens = s.str.len()
            mean_len = lens.mean()
            if pd.isna(mean_len):
                mean_len = 0.0
            avg_len[col] = float(mean_len)

            counts = s.value_counts(dropna=False)
            if N > 0 and len(counts) > 0:
                freqs = counts.values.astype("float64")
                probs = freqs / N
                p = float((probs * probs).sum())
            else:
                p = 1.0
            p_match[col] = p

        p_arr = [p_match[c] for c in cols]
        l_arr = [avg_len[c] for c in cols]

        m = n_cols
        max_bruteforce_cols = 10

        if m <= max_bruteforce_cols:
            indices = list(range(m))
            best_perm_idx = tuple(indices)
            best_score = -1.0

            for perm in itertools.permutations(indices, m):
                prod = 1.0
                score = 0.0
                for idx in perm:
                    prod *= p_arr[idx]
                    score += l_arr[idx] * prod
                if score > best_score:
                    best_score = score
                    best_perm_idx = perm

            new_cols = [cols[i] for i in best_perm_idx]
            return work_df[new_cols]

        sort_keys = [(-p_match[c], -avg_len[c], i) for i, c in enumerate(cols)]
        sort_keys.sort()
        new_cols = [cols[i] for _, _, i in sort_keys]
        return work_df[new_cols]