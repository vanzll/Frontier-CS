import numpy as np
import pandas as pd
from typing import List, Any, Optional


def _concat_columns_as_strings(df: pd.DataFrame, cols: List[str]) -> pd.Series:
    if len(cols) == 1:
        return df[cols[0]].astype(str)
    arr = df[cols].astype(str).to_numpy(dtype=str, copy=False)
    out = arr[:, 0]
    for j in range(1, arr.shape[1]):
        out = np.char.add(out, arr[:, j])
    return pd.Series(out, index=df.index)


def _apply_column_merges(df: pd.DataFrame, col_merge: Optional[list]) -> pd.DataFrame:
    if not col_merge:
        return df

    df2 = df.copy(deep=False)
    orig_cols = list(df2.columns)
    used = set()

    for group in col_merge:
        if not group:
            continue

        cols = []
        for item in group:
            name = None
            if isinstance(item, (int, np.integer)):
                idx = int(item)
                if 0 <= idx < len(orig_cols):
                    name = orig_cols[idx]
            else:
                name = str(item)

            if name is None:
                continue
            if name in df2.columns and name not in used:
                cols.append(name)

        if len(cols) <= 1:
            continue

        for c in cols:
            used.add(c)

        base_name = "__merge__" + "+".join(cols)
        new_name = base_name
        k = 1
        while new_name in df2.columns:
            new_name = f"{base_name}_{k}"
            k += 1

        loc = min(int(df2.columns.get_loc(c)) for c in cols)
        merged = _concat_columns_as_strings(df2, cols)

        df2 = df2.drop(columns=cols)
        loc = min(loc, len(df2.columns))
        df2.insert(loc, new_name, merged)

    return df2


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
        if not isinstance(df, pd.DataFrame):
            return df

        dfm = _apply_column_merges(df, col_merge)
        cols = list(dfm.columns)
        m = len(cols)
        n = len(dfm)
        if m <= 1 or n <= 1:
            return dfm

        K_cap = 6000
        K = min(n, K_cap, int(early_stop) if early_stop is not None else K_cap)
        if K <= 0:
            return dfm

        if n > K:
            rng = np.random.default_rng(0)
            idx = rng.choice(n, size=K, replace=False)
            sample = dfm.iloc[idx]
        else:
            sample = dfm

        K = len(sample)
        if K <= 1:
            return dfm

        sample_str = sample.astype(str)
        sample_np = sample_str.to_numpy(dtype=object, copy=False)

        codes_u32: List[np.ndarray] = [None] * m
        codes_u64: List[np.ndarray] = [None] * m
        lens_list: List[np.ndarray] = [None] * m
        total_len_list = np.zeros(m, dtype=np.int64)
        avg_len_list = np.zeros(m, dtype=np.float64)

        for j in range(m):
            col_vals = sample_np[:, j]
            codes, uniques = pd.factorize(col_vals, sort=False)
            if codes.dtype != np.int64:
                codes = codes.astype(np.int64, copy=False)
            cu32 = codes.astype(np.uint32, copy=False)
            codes_u32[j] = cu32
            codes_u64[j] = cu32.astype(np.uint64, copy=False)

            lens = np.fromiter((len(u) for u in uniques), dtype=np.int32, count=len(uniques))
            lens_list[j] = lens

            tot = lens.take(cu32).sum(dtype=np.int64)
            total_len_list[j] = tot
            avg_len_list[j] = float(tot) / float(K)

        group_id = np.zeros(K, dtype=np.uint32)

        remaining = list(range(m))
        order_idx: List[int] = []
        remaining_avg_len = float(avg_len_list.sum())

        tmp = np.empty(K, dtype=np.uint64)

        alpha = 0.3

        while remaining:
            C = int(group_id.max()) + 1
            if C >= K:
                order_idx.extend(remaining)
                break

            gid_shifted = (group_id.astype(np.uint64, copy=False) << 32)

            best_score = -1e300
            best_j = None

            for j in remaining:
                np.bitwise_or(gid_shifted, codes_u64[j], out=tmp)
                uniq, first_idx = np.unique(tmp, return_index=True)
                G = int(uniq.size)

                lens = lens_list[j]
                cu32 = codes_u32[j]
                sum_unique_len = lens.take(cu32.take(first_idx)).sum(dtype=np.int64)

                benefit = int(total_len_list[j] - sum_unique_len)

                future_len = remaining_avg_len - float(avg_len_list[j])
                inc_groups = G - C
                penalty = alpha * float(inc_groups) * future_len

                score = float(benefit) - penalty

                if score > best_score:
                    best_score = score
                    best_j = j

            if best_j is None:
                order_idx.extend(remaining)
                break

            order_idx.append(best_j)
            remaining.remove(best_j)
            remaining_avg_len -= float(avg_len_list[best_j])

            np.bitwise_or(gid_shifted, codes_u64[best_j], out=tmp)
            _, inv = np.unique(tmp, return_inverse=True)
            group_id = inv.astype(np.uint32, copy=False)

        ordered_cols = [cols[i] for i in order_idx]
        return dfm.loc[:, ordered_cols]