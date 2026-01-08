import pandas as pd
import numpy as np


class Solution:
    def _resolve_group(self, df: pd.DataFrame, group) -> list:
        cols = list(df.columns)
        resolved = []
        for spec in group:
            if isinstance(spec, int):
                if 0 <= spec < len(cols):
                    name = cols[spec]
                    if name not in resolved:
                        resolved.append(name)
            else:
                if spec in df.columns and spec not in resolved:
                    resolved.append(spec)
        return resolved

    def _apply_col_merge_inplace(self, df: pd.DataFrame, col_merge: list) -> None:
        if not col_merge:
            return
        for group in col_merge:
            if not group:
                continue
            if isinstance(group, (str, bytes)):
                continue
            group_cols = self._resolve_group(df, group)
            if len(group_cols) < 2:
                continue
            if not all(c in df.columns for c in group_cols):
                continue
            cols_list = list(df.columns)
            try:
                first_idx = min(cols_list.index(c) for c in group_cols)
            except ValueError:
                continue
            base_col = group_cols[0]
            merged_series = df[base_col].astype(str)
            for c in group_cols[1:]:
                merged_series = merged_series + df[c].astype(str)
            df.drop(columns=group_cols, inplace=True)
            df.insert(first_idx, base_col, merged_series)

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
            try:
                self._apply_col_merge_inplace(df_work, col_merge)
            except Exception:
                pass

        n_rows, n_cols = df_work.shape
        if n_rows <= 1 or n_cols <= 1:
            return df_work

        thr = distinct_value_threshold if distinct_value_threshold is not None else 0.0
        cols = list(df_work.columns)
        idx_map = {col: i for i, col in enumerate(cols)}
        scores = {}

        for col in cols:
            s = df_work[col].astype(str)
            N = len(s)
            if N == 0:
                scores[col] = 0.0
                continue

            lens = s.str.len()
            if len(lens) == 0:
                mean_len = 0.0
            else:
                mean_len = float(lens.mean())

            try:
                distinct_cnt = s.nunique(dropna=False)
            except Exception:
                distinct_cnt = s.astype(object).nunique(dropna=False)

            if N > 0:
                distinct_ratio = float(distinct_cnt) / float(N)
            else:
                distinct_ratio = 0.0

            if thr > 0.0:
                if distinct_ratio < thr:
                    char_stability = (thr - distinct_ratio) / thr
                    if char_stability < 0.0:
                        char_stability = 0.0
                    elif char_stability > 1.0:
                        char_stability = 1.0
                else:
                    char_stability = 0.0
            else:
                char_stability = max(0.0, 1.0 - distinct_ratio)

            arr = s.to_numpy()
            if N > 1:
                try:
                    eq = (arr[1:] == arr[:-1])
                    equals_adj = float(np.sum(eq)) / float(N - 1)
                except Exception:
                    equals_adj = 0.0
            else:
                equals_adj = 0.0

            combined = 0.6 * char_stability + 0.4 * equals_adj
            if combined < 0.0:
                combined = 0.0
            score = mean_len * combined
            scores[col] = float(score)

        ordered_cols = sorted(cols, key=lambda c: (-scores.get(c, 0.0), idx_map[c]))
        df_reordered = df_work[ordered_cols]
        return df_reordered