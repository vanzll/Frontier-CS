import pandas as pd
import numpy as np
from typing import List, Any, Dict, Tuple, Optional
import heapq


class Solution:
    def __init__(self):
        self._rng = np.random.default_rng(0)

    @staticmethod
    def _normalize_col_merge_indices(col_merge: list, cols: List[str]) -> list:
        if not col_merge:
            return col_merge
        ints = []
        for g in col_merge:
            if isinstance(g, (list, tuple)):
                for x in g:
                    if isinstance(x, (int, np.integer)):
                        ints.append(int(x))
        if not ints:
            return col_merge

        m = len(cols)
        if any(i == 0 for i in ints):
            mode = "zero"
        else:
            mx = max(ints)
            if mx == m:
                mode = "one"
            elif mx <= m - 1:
                mode = "zero"
            else:
                mode = "one"
        return [("IDXMODE", mode)] + col_merge

    @staticmethod
    def _map_merge_item(x: Any, cols: List[str], idx_mode: str) -> Optional[str]:
        if isinstance(x, str):
            return x
        if isinstance(x, (int, np.integer)):
            i = int(x)
            if idx_mode == "one":
                i -= 1
            if 0 <= i < len(cols):
                return cols[i]
        return None

    def _apply_col_merge(self, df: pd.DataFrame, col_merge: list) -> pd.DataFrame:
        if not col_merge:
            return df

        cols0 = list(df.columns)
        col_merge = self._normalize_col_merge_indices(col_merge, cols0)
        idx_mode = "zero"
        if col_merge and isinstance(col_merge[0], tuple) and col_merge[0][0] == "IDXMODE":
            idx_mode = col_merge[0][1]
            col_merge = col_merge[1:]

        out = df.copy(deep=False)
        for group in col_merge:
            if not isinstance(group, (list, tuple)) or len(group) < 2:
                continue

            current_cols = list(out.columns)
            mapped = []
            for item in group:
                name = self._map_merge_item(item, cols0, idx_mode)
                if name is None:
                    continue
                if name in out.columns and name not in mapped:
                    mapped.append(name)

            if len(mapped) < 2:
                continue

            base_name = "|".join(mapped)
            new_name = base_name
            k = 1
            while new_name in out.columns:
                new_name = f"{base_name}__m{k}"
                k += 1

            s = out[mapped[0]].astype(str)
            if len(mapped) > 1:
                others = [out[c].astype(str) for c in mapped[1:]]
                s = s.str.cat(others, sep="")
            out[new_name] = s
            out = out.drop(columns=mapped)

        return out

    @staticmethod
    def _col_stats_from_freq(freq: Dict[str, int], n: int) -> Tuple[float, float, float, float]:
        if n <= 1:
            distinct_ratio = (len(freq) / n) if n else 0.0
            total_len = 0
            for s, c in freq.items():
                total_len += len(s) * c
            avg_len = (total_len / n) if n else 0.0
            return 0.0, 0.0, distinct_ratio, avg_len

        denom_pairs2 = n * (n - 1)
        eq_sum2 = 0
        total_len = 0
        for s, c in freq.items():
            eq_sum2 += c * (c - 1)
            total_len += len(s) * c
        q = eq_sum2 / denom_pairs2
        distinct_ratio = len(freq) / n
        avg_len = total_len / n

        children = [{}]
        counts = [0]

        for s, mult in freq.items():
            node = 0
            counts[0] += mult
            for ch in s:
                nxt = children[node].get(ch)
                if nxt is None:
                    nxt = len(children)
                    children[node][ch] = nxt
                    children.append({})
                    counts.append(0)
                node = nxt
                counts[node] += mult

        total_pairs = n * (n - 1) // 2
        sum_pairs = 0
        for c in counts[1:]:
            if c >= 2:
                sum_pairs += (c * (c - 1)) // 2
        e = (sum_pairs / total_pairs) if total_pairs > 0 else 0.0

        return q, e, distinct_ratio, avg_len

    @staticmethod
    def _sort_key(q: float, e: float, distinct_ratio: float, avg_len: float, distinct_value_threshold: float) -> float:
        if avg_len <= 0.0:
            return -1e30
        if q >= 0.999999999999:
            return 1e30 + e
        key = e / (1.0 - q)
        if distinct_ratio > distinct_value_threshold:
            key *= 1.0 / (1.0 + 2.0 * (distinct_ratio - distinct_value_threshold))
        return key

    @staticmethod
    def _apply_deps(initial_order: List[str], one_way_dep: list, available_cols) -> List[str]:
        cols_set = set(initial_order)
        pos = {c: i for i, c in enumerate(initial_order)}

        edges = []
        for dep in (one_way_dep or []):
            if isinstance(dep, (list, tuple)) and len(dep) >= 2:
                a, b = dep[0], dep[1]
                if isinstance(a, (int, np.integer)):
                    ia = int(a)
                    if 0 <= ia < len(available_cols):
                        a = list(available_cols)[ia]
                    elif 1 <= ia <= len(available_cols):
                        a = list(available_cols)[ia - 1]
                    else:
                        continue
                if isinstance(b, (int, np.integer)):
                    ib = int(b)
                    if 0 <= ib < len(available_cols):
                        b = list(available_cols)[ib]
                    elif 1 <= ib <= len(available_cols):
                        b = list(available_cols)[ib - 1]
                    else:
                        continue
                if isinstance(a, str) and isinstance(b, str) and a in cols_set and b in cols_set and a != b:
                    edges.append((a, b))

        if not edges:
            return initial_order

        g = {c: [] for c in initial_order}
        indeg = {c: 0 for c in initial_order}
        for a, b in edges:
            g[a].append(b)
            indeg[b] += 1

        heap = []
        for c in initial_order:
            if indeg[c] == 0:
                heapq.heappush(heap, (pos[c], c))

        out = []
        while heap:
            _, c = heapq.heappop(heap)
            out.append(c)
            for nb in g[c]:
                indeg[nb] -= 1
                if indeg[nb] == 0:
                    heapq.heappush(heap, (pos[nb], nb))

        if len(out) != len(initial_order):
            return initial_order
        return out

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
        if df is None or df.shape[1] <= 1:
            return df

        df2 = self._apply_col_merge(df, col_merge)

        n = len(df2)
        if n <= 1 or df2.shape[1] <= 1:
            return df2

        max_sample = 8000
        min_sample = 2000
        if early_stop is None or early_stop <= 0:
            sample_n = min(n, max_sample)
        else:
            sample_n = min(n, int(early_stop), max_sample)
        sample_n = max(1, min(n, max(min_sample, sample_n)))

        if n > sample_n:
            idx = self._rng.choice(n, size=sample_n, replace=False)
            df_s = df2.iloc[idx]
        else:
            df_s = df2

        df_s = df_s.astype(str)

        cols = list(df2.columns)
        metrics = []
        for col in cols:
            arr = df_s[col].to_numpy(dtype=object, copy=False)
            freq: Dict[str, int] = {}
            for s in arr:
                freq[s] = freq.get(s, 0) + 1
            q, e, distinct_ratio, avg_len = self._col_stats_from_freq(freq, len(arr))
            key = self._sort_key(q, e, distinct_ratio, avg_len, distinct_value_threshold)
            metrics.append((key, e, q, -distinct_ratio, avg_len, col))

        metrics.sort(reverse=True)
        ordered = [x[-1] for x in metrics]

        if one_way_dep:
            ordered = self._apply_deps(ordered, one_way_dep, df2.columns)

        return df2.loc[:, ordered]