import pandas as pd
import numpy as np


class Solution:
    def _apply_col_merge(self, df: pd.DataFrame, col_merge):
        if not col_merge:
            return df

        # Use original columns to resolve integer indices
        orig_cols = list(df.columns)
        groups = []
        for grp in col_merge:
            if not grp:
                continue
            names = []
            for item in grp:
                if isinstance(item, int):
                    if 0 <= item < len(orig_cols):
                        name = orig_cols[item]
                        if name not in names:
                            names.append(name)
                else:
                    name_str = str(item)
                    if name_str not in names:
                        names.append(name_str)
            if names:
                groups.append(names)

        for names in groups:
            # Only merge columns that still exist in df
            available = [c for c in names if c in df.columns]
            if len(available) <= 1:
                continue

            base_name = "MERGED_" + "_".join(available)
            new_name = base_name
            if new_name in df.columns:
                suffix = 1
                while f"{base_name}_{suffix}" in df.columns:
                    suffix += 1
                new_name = f"{base_name}_{suffix}"

            # Build merged column by concatenating string representations without separators
            merged_series = df[available[0]].astype(str)
            for c in available[1:]:
                merged_series = merged_series.str.cat(df[c].astype(str), na_rep='')

            first_pos = df.columns.get_loc(available[0])
            df.insert(first_pos, new_name, merged_series)
            df = df.drop(columns=available)

        return df

    def _compute_column_score(
        self,
        values: np.ndarray,
        orig_series: pd.Series,
        n_rows: int,
        distinct_value_threshold: float,
    ) -> float:
        if n_rows == 0:
            return 0.0

        try:
            nunique = orig_series.nunique(dropna=False)
        except Exception:
            nunique = pd.Series(orig_series).nunique(dropna=False)

        if n_rows > 0:
            distinct_ratio = float(nunique) / float(n_rows)
        else:
            distinct_ratio = 1.0

        if distinct_ratio < 0.0:
            distinct_ratio = 0.0
        elif distinct_ratio > 1.0:
            distinct_ratio = 1.0

        total_len = 0
        total_lcp_prev = 0

        prev = values[0]
        prev_len = len(prev)
        total_len += prev_len

        n = n_rows
        for idx in range(1, n):
            cur = values[idx]
            cur_len = len(cur)
            total_len += cur_len

            if prev_len and cur_len:
                if prev_len < cur_len:
                    min_len = prev_len
                else:
                    min_len = cur_len
                k = 0
                s1 = prev
                s2 = cur
                while k < min_len and s1[k] == s2[k]:
                    k += 1
                total_lcp_prev += k

            prev = cur
            prev_len = cur_len

        avg_len = float(total_len) / float(n_rows)
        if n_rows > 1:
            avg_lcp_prev = float(total_lcp_prev) / float(n_rows - 1)
        else:
            avg_lcp_prev = 0.0

        if avg_len > 0.0:
            ratio_prev = avg_lcp_prev / avg_len
        else:
            ratio_prev = 0.0

        if ratio_prev < 0.0:
            ratio_prev = 0.0
        elif ratio_prev > 1.0:
            ratio_prev = 1.0

        if distinct_value_threshold is not None and distinct_value_threshold > 0.0:
            low_card_factor = (distinct_value_threshold - distinct_ratio) / distinct_value_threshold
            if low_card_factor < 0.0:
                low_card_factor = 0.0
            elif low_card_factor > 1.0:
                low_card_factor = 1.0
        else:
            low_card_factor = 1.0 - distinct_ratio
            if low_card_factor < 0.0:
                low_card_factor = 0.0
            elif low_card_factor > 1.0:
                low_card_factor = 1.0

        predictability = 0.7 * ratio_prev + 0.3 * low_card_factor
        if predictability < 0.0:
            predictability = 0.0
        elif predictability > 1.0:
            predictability = 1.0

        score = avg_len * predictability
        return float(score)

    def _approx_total_lcp(
        self,
        col_strings: dict,
        columns,
        n_rows: int,
        row_limit: int,
    ) -> int:
        if n_rows <= 1:
            return 0

        if row_limit is None or row_limit <= 0 or row_limit > n_rows:
            limit = n_rows
        else:
            limit = row_limit

        if limit <= 1:
            return 0

        total_lcp = 0
        cols = columns
        col_vals = col_strings

        for i in range(1, limit):
            lcp = 0
            for col in cols:
                arr = col_vals[col]
                s1 = arr[i - 1]
                s2 = arr[i]
                len1 = len(s1)
                len2 = len(s2)

                if len1 == 0 and len2 == 0:
                    continue

                if len1 < len2:
                    min_len = len1
                else:
                    min_len = len2

                k = 0
                while k < min_len and s1[k] == s2[k]:
                    k += 1

                lcp += k
                if k < len1 or k < len2:
                    break

            total_lcp += lcp

        return total_lcp

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
        df_work = df.copy()

        if col_merge:
            df_work = self._apply_col_merge(df_work, col_merge)

        if df_work.shape[1] <= 1:
            return df_work

        n_rows = len(df_work)
        cols = list(df_work.columns)

        col_strings = {}
        for col in cols:
            col_strings[col] = df_work[col].astype(str).to_numpy(dtype=object, copy=False)

        scores = {}
        for col in cols:
            values = col_strings[col]
            orig_series = df_work[col]
            scores[col] = self._compute_column_score(
                values,
                orig_series,
                n_rows,
                distinct_value_threshold,
            )

        baseline_order = tuple(cols)

        if isinstance(early_stop, int) and early_stop > 0:
            row_limit = early_stop if n_rows > early_stop else n_rows
        else:
            row_limit = n_rows

        baseline_lcp = self._approx_total_lcp(col_strings, baseline_order, n_rows, row_limit)

        col_index = {c: i for i, c in enumerate(cols)}
        sorted_cols = sorted(
            cols,
            key=lambda c: (-scores.get(c, 0.0), col_index[c]),
        )
        candidate_order = tuple(sorted_cols)

        if candidate_order == baseline_order:
            final_order = baseline_order
        else:
            candidate_lcp = self._approx_total_lcp(col_strings, candidate_order, n_rows, row_limit)
            if candidate_lcp > baseline_lcp:
                final_order = candidate_order
            else:
                final_order = baseline_order

        df_out = df_work.loc[:, list(final_order)]
        return df_out