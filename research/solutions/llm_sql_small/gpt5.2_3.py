import pandas as pd
import numpy as np
from typing import List, Optional, Any, Tuple


class Solution:
    def __init__(self):
        self._fnv_offset = 1469598103934665603
        self._fnv_prime = 1099511628211
        self._mask64 = (1 << 64) - 1

    def _resolve_merge_groups(self, df: pd.DataFrame, col_merge: Optional[list]) -> List[List[str]]:
        if not col_merge:
            return []
        orig_cols = list(df.columns)

        def resolve_elem(e: Any) -> str:
            if isinstance(e, (int, np.integer)):
                idx = int(e)
                if 0 <= idx < len(orig_cols):
                    return str(orig_cols[idx])
                return str(e)
            return str(e)

        groups = []
        for g in col_merge:
            if g is None:
                continue
            if isinstance(g, (str, int, np.integer)):
                groups.append([resolve_elem(g)])
                continue
            try:
                lst = [resolve_elem(x) for x in list(g)]
            except Exception:
                lst = [resolve_elem(g)]
            if lst:
                groups.append(lst)
        return groups

    def _apply_col_merge(self, df: pd.DataFrame, col_merge: Optional[list]) -> pd.DataFrame:
        groups = self._resolve_merge_groups(df, col_merge)
        if not groups:
            return df

        dfw = df.copy()
        for group in groups:
            cols = [c for c in group if c in dfw.columns]
            if len(cols) <= 1:
                continue

            merged = dfw[cols[0]].astype(str)
            for c in cols[1:]:
                merged = merged + dfw[c].astype(str)

            base_name = "__MERGED__" + "_".join(map(str, cols))
            new_name = base_name
            k = 1
            while new_name in dfw.columns:
                k += 1
                new_name = f"{base_name}__{k}"

            insert_pos = int(dfw.columns.get_loc(cols[0]))
            dfw.insert(insert_pos, new_name, merged)
            dfw.drop(columns=cols, inplace=True)

        return dfw

    def _sample_indices(self, n: int, k: int) -> List[int]:
        if n <= k:
            return list(range(n))
        step = max(1, n // k)
        idx = list(range(0, n, step))
        if len(idx) > k:
            idx = idx[:k]
        if idx and idx[-1] != n - 1 and len(idx) < k:
            idx.append(n - 1)
        return idx

    def _col_stats_and_sample(
        self,
        s: pd.Series,
        sample_idx: List[int],
        lcap: int,
    ) -> Tuple[float, float, float, float, List[str]]:
        ss = s.astype(str)
        arr = ss.to_numpy(dtype=object, copy=False)
        sample = [arr[i] for i in sample_idx]

        n = len(ss)
        if n <= 1:
            return 0.0, 1.0, float(len(sample[0]) if sample else 0), 0.0, sample

        vc = ss.value_counts(dropna=False)
        freqs = vc.to_numpy(dtype=np.int64, copy=False)
        values = vc.index.to_list()

        n2 = float(n * n)
        peq = float(np.dot(freqs, freqs)) / n2
        distinct_ratio = float(len(values)) / float(n)

        lens = np.fromiter((len(v) for v in values), dtype=np.int64, count=len(values))
        avg_len = float(np.dot(freqs, lens)) / float(n)

        # Approximate expected LCP length within this column:
        # E[lcp] = sum_{t>=1} P(lcp >= t) = sum_t sum_prefix (cnt_prefix/N)^2
        # Truncate prefixes at lcap and add exact tail for equal strings beyond lcap.
        prefix_dicts = [dict() for _ in range(lcap)]
        fnv_offset = self._fnv_offset
        fnv_prime = self._fnv_prime
        mask = self._mask64

        extra_tail = 0
        for v, f, L in zip(values, freqs, lens):
            if L > lcap:
                extra_tail += int(f) * int(f) * (int(L) - lcap)
            h = fnv_offset
            upto = int(L) if int(L) < lcap else lcap
            for i in range(upto):
                h ^= ord(v[i])
                h = (h * fnv_prime) & mask
                d = prefix_dicts[i]
                prev = d.get(h)
                if prev is None:
                    d[h] = int(f)
                else:
                    d[h] = prev + int(f)

        inv_n2 = 1.0 / n2
        e_trunc = 0.0
        for d in prefix_dicts:
            if not d:
                continue
            ssq = 0
            for c in d.values():
                ssq += c * c
            e_trunc += ssq * inv_n2

        e_lcp = e_trunc + (extra_tail * inv_n2)
        return e_lcp, peq, avg_len, distinct_ratio, sample

    def _score_perm(self, col_samples: List[List[str]], perm: List[int]) -> int:
        # Trie via a single dict of edges keyed by (node<<21 | codepoint).
        # Depth counted in characters. Score is sum of hit lengths for rows i>=2.
        edges = {}
        node_count = 1
        total = 0
        mask_code = (1 << 21) - 1

        K = len(col_samples[0]) if col_samples else 0
        for r in range(K):
            node = 0
            depth = 0
            miss = False
            for ci in perm:
                s = col_samples[ci][r]
                if not s:
                    continue
                if miss:
                    for ch in s:
                        code = ord(ch) & mask_code
                        key = (node << 21) | code
                        edges[key] = node_count
                        node = node_count
                        node_count += 1
                else:
                    for ch in s:
                        code = ord(ch) & mask_code
                        key = (node << 21) | code
                        nxt = edges.get(key)
                        if nxt is None:
                            miss = True
                            edges[key] = node_count
                            node = node_count
                            node_count += 1
                        else:
                            depth += 1
                            node = nxt
            if r > 0:
                total += depth
        return total

    def _hill_climb(self, col_samples: List[List[str]], start_perm: List[int], max_iters: int = 10) -> Tuple[List[int], int]:
        m = len(start_perm)
        best = list(start_perm)
        best_score = self._score_perm(col_samples, best)

        for _ in range(max_iters):
            improved = False
            # First-improvement over all swaps
            for i in range(m - 1):
                for j in range(i + 1, m):
                    cand = best[:]
                    cand[i], cand[j] = cand[j], cand[i]
                    sc = self._score_perm(col_samples, cand)
                    if sc > best_score:
                        best_score = sc
                        best = cand
                        improved = True
                        break
                if improved:
                    break
            if not improved:
                break
        return best, best_score

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
        dfw = self._apply_col_merge(df, col_merge)

        cols = list(dfw.columns)
        m = len(cols)
        if m <= 1:
            return dfw

        n = len(dfw)
        if n <= 1:
            return dfw

        K = int(min(n, max(1500, min(5000, early_stop if early_stop is not None else 5000))))
        sample_idx = self._sample_indices(n, K)
        K = len(sample_idx)

        # Prefix cap for stats (not scoring)
        lcap = 16

        col_samples: List[List[str]] = []
        stats = []
        for c in cols:
            e_lcp, peq, avg_len, distinct_ratio, sample = self._col_stats_and_sample(dfw[c], sample_idx, lcap)
            col_samples.append(sample)
            stats.append((e_lcp, peq, avg_len, distinct_ratio))

        # Build heuristic perms
        def key_ratio(e_lcp: float, peq: float, distinct_ratio: float) -> float:
            if peq >= 1.0 - 1e-15:
                return float("inf")
            k = e_lcp / max(1e-15, (1.0 - peq))
            if distinct_ratio > distinct_value_threshold:
                k *= 0.7
            return k

        idxs = list(range(m))
        perm_ratio = sorted(
            idxs,
            key=lambda i: (
                -key_ratio(stats[i][0], stats[i][1], stats[i][3]),
                stats[i][3],
                -stats[i][2],
            ),
        )

        perm_distinct = sorted(
            idxs,
            key=lambda i: (
                stats[i][3],
                -stats[i][1],
                -stats[i][2],
            ),
        )

        perm_orig = idxs[:]

        # Pick best start among a small set
        starts = [perm_ratio, perm_distinct, perm_orig]
        best_start = starts[0]
        best_start_score = self._score_perm(col_samples, best_start)
        for p in starts[1:]:
            sc = self._score_perm(col_samples, p)
            if sc > best_start_score:
                best_start_score = sc
                best_start = p

        best_perm, _ = self._hill_climb(col_samples, best_start, max_iters=10)

        ordered_cols = [cols[i] for i in best_perm]
        return dfw.loc[:, ordered_cols]