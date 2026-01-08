import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Any, Optional


class Solution:
    def _resolve_merge_groups(self, df: pd.DataFrame, col_merge: list) -> List[List[str]]:
        if not col_merge:
            return []
        orig_cols = list(df.columns)
        groups: List[List[str]] = []
        for g in col_merge:
            if g is None:
                continue
            if isinstance(g, (str, int)):
                g = [g]
            names: List[str] = []
            for x in g:
                if isinstance(x, int):
                    if 0 <= x < len(orig_cols):
                        names.append(orig_cols[x])
                else:
                    names.append(str(x))
            # dedup while preserving order
            seen = set()
            names2 = []
            for n in names:
                if n not in seen:
                    seen.add(n)
                    names2.append(n)
            if len(names2) >= 2:
                groups.append(names2)
        return groups

    def _apply_merges(self, df: pd.DataFrame, groups: List[List[str]]) -> pd.DataFrame:
        if not groups:
            return df
        df2 = df.copy()
        existing = set(df2.columns)
        used = set()
        for group in groups:
            group_names = [c for c in group if c in existing and c not in used]
            if len(group_names) < 2:
                continue
            merged_name_base = "__".join(group_names)
            merged_name = merged_name_base
            k = 1
            while merged_name in df2.columns:
                merged_name = f"{merged_name_base}__m{k}"
                k += 1

            s = df2[group_names[0]].astype(str)
            for c in group_names[1:]:
                s = s + df2[c].astype(str)
            df2[merged_name] = s
            df2.drop(columns=group_names, inplace=True, errors="ignore")

            existing = set(df2.columns)
            for c in group_names:
                used.add(c)
        return df2

    def _enforce_one_way_deps(self, col_names: List[str], perm_idx: List[int], one_way_dep: Optional[list]) -> List[int]:
        if not one_way_dep:
            return perm_idx
        name_to_pos = {col_names[i]: i for i in range(len(col_names))}
        edges: List[Tuple[int, int]] = []
        for dep in one_way_dep:
            if not dep:
                continue
            if isinstance(dep, (list, tuple)) and len(dep) >= 2:
                a, b = dep[0], dep[1]
            else:
                continue
            a = str(a)
            b = str(b)
            if a in name_to_pos and b in name_to_pos:
                edges.append((name_to_pos[a], name_to_pos[b]))
        if not edges:
            return perm_idx

        perm = perm_idx[:]
        pos_in_perm = {v: i for i, v in enumerate(perm)}
        changed = True
        it = 0
        while changed and it < 20:
            it += 1
            changed = False
            for a, b in edges:
                pa = pos_in_perm.get(a, None)
                pb = pos_in_perm.get(b, None)
                if pa is None or pb is None:
                    continue
                if pa > pb:
                    # move a before b
                    val = perm.pop(pa)
                    pb = pos_in_perm[b]
                    perm.insert(pb, val)
                    pos_in_perm = {v: i for i, v in enumerate(perm)}
                    changed = True
        return perm

    def _prepare_sample_codes_and_lens(
        self, df: pd.DataFrame, cols: List[str], K: int
    ) -> Tuple[List[List[int]], List[List[int]], List[Tuple[float, float]]]:
        # Returns:
        #   codes_list: per col list of int codes (>=1) length K
        #   lens_list: per col list of int lengths length K
        #   stats: per col (distinct_ratio, avg_len)
        codes_list: List[List[int]] = []
        lens_list: List[List[int]] = []
        stats: List[Tuple[float, float]] = []

        for c in cols:
            arr = df[c].astype(str).to_numpy()[:K]
            raw_codes = pd.factorize(arr, sort=False)[0].astype(np.int32)
            if K > 0:
                uniq = int(raw_codes.max()) + 1
                distinct_ratio = float(uniq) / float(K)
            else:
                distinct_ratio = 1.0
            lens = [len(x) for x in arr]
            avg_len = float(sum(lens)) / float(K) if K > 0 else 0.0
            codes = (raw_codes.astype(np.int64) + 1).tolist()
            codes_list.append(codes)
            lens_list.append(lens)
            stats.append((distinct_ratio, avg_len))
        return codes_list, lens_list, stats

    def _score_perm(
        self,
        perm: Tuple[int, ...],
        codes_list: List[List[int]],
        lens_list: List[List[int]],
        K: int,
        cache: Dict[Tuple[int, ...], int],
    ) -> int:
        if perm in cache:
            return cache[perm]
        m = len(perm)
        if m <= 1 or K <= 1:
            cache[perm] = 0
            return 0

        # Rolling hash of column-codes for prefixes
        P = 11400714819323198485  # 64-bit odd constant
        MASK = (1 << 64) - 1

        seen = [set() for _ in range(m)]
        keys = [0] * m
        cumlens = [0] * m

        total = 0
        for r in range(K):
            key = 1469598103934665603  # offset basis
            cum = 0
            for t in range(m):
                col = perm[t]
                key = (key * P + codes_list[col][r]) & MASK
                keys[t] = key
                cum += lens_list[col][r]
                cumlens[t] = cum

            best = -1
            for t in range(m - 1, -1, -1):
                if keys[t] in seen[t]:
                    best = t
                    break
            if best >= 0:
                total += cumlens[best]

            for t in range(m):
                seen[t].add(keys[t])

        cache[perm] = total
        return total

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

        groups = self._resolve_merge_groups(df, col_merge)
        df2 = self._apply_merges(df, groups)

        cols = list(df2.columns)
        m = len(cols)
        if m <= 1:
            return df2

        N = len(df2)
        if N <= 1:
            return df2

        K = min(N, 12000)
        codes_list, lens_list, stats = self._prepare_sample_codes_and_lens(df2, cols, K)

        # Heuristic order
        def hkey(i: int) -> Tuple[float, float, float, int]:
            distinct_ratio, avg_len = stats[i]
            bump = 1.0 if distinct_ratio >= distinct_value_threshold else 0.0
            return (distinct_ratio + bump, distinct_ratio, -avg_len, i)

        base_order = tuple(sorted(range(m), key=hkey))

        cache: Dict[Tuple[int, ...], int] = {}

        # Greedy construction
        remaining = list(base_order)
        current: List[int] = []
        for _ in range(m):
            best_c = None
            best_score = -1
            # sort remaining by heuristic for deterministic trials
            rem_sorted = sorted(remaining, key=hkey)
            for c in rem_sorted:
                rest = [x for x in rem_sorted if x != c]
                trial = tuple(current + [c] + rest)
                s = self._score_perm(trial, codes_list, lens_list, K, cache)
                if s > best_score:
                    best_score = s
                    best_c = c
            current.append(best_c)
            remaining.remove(best_c)

        greedy_perm = tuple(current)
        heuristic_perm = base_order

        best_perm = greedy_perm
        best_score = self._score_perm(best_perm, codes_list, lens_list, K, cache)
        hs = self._score_perm(heuristic_perm, codes_list, lens_list, K, cache)
        if hs > best_score:
            best_perm = heuristic_perm
            best_score = hs

        # Local search: swaps + insertions
        perm = list(best_perm)
        for _it in range(4):
            base = self._score_perm(tuple(perm), codes_list, lens_list, K, cache)
            best_local = base
            best_local_perm = None

            # swaps
            for i in range(m):
                for j in range(i + 1, m):
                    trial = perm[:]
                    trial[i], trial[j] = trial[j], trial[i]
                    s = self._score_perm(tuple(trial), codes_list, lens_list, K, cache)
                    if s > best_local:
                        best_local = s
                        best_local_perm = trial

            # insertions
            for i in range(m):
                for j in range(m):
                    if i == j:
                        continue
                    trial = perm[:]
                    v = trial.pop(i)
                    trial.insert(j, v)
                    s = self._score_perm(tuple(trial), codes_list, lens_list, K, cache)
                    if s > best_local:
                        best_local = s
                        best_local_perm = trial

            if best_local_perm is None:
                break
            perm = best_local_perm

        perm = self._enforce_one_way_deps(cols, perm, one_way_dep)
        ordered_cols = [cols[i] for i in perm]
        return df2.loc[:, ordered_cols]