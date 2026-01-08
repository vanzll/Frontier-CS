import pandas as pd
import numpy as np
import random
from typing import List, Optional, Any, Tuple


class Solution:
    def _normalize_merge_groups(self, df: pd.DataFrame, col_merge: Optional[list]) -> List[List[Any]]:
        if not col_merge:
            return []
        cols = list(df.columns)
        col_set = set(cols)
        groups: List[List[Any]] = []
        for g in col_merge:
            if not g:
                continue
            labels: List[Any] = []
            for x in g:
                if isinstance(x, (int, np.integer)):
                    xi = int(x)
                    if 0 <= xi < len(cols):
                        labels.append(cols[xi])
                else:
                    if x in col_set:
                        labels.append(x)
            if len(labels) < 2:
                continue
            seen = set()
            dedup = []
            for c in labels:
                if c not in seen:
                    seen.add(c)
                    dedup.append(c)
            if len(dedup) >= 2:
                groups.append(dedup)
        return groups

    def _apply_col_merge(self, df: pd.DataFrame, col_merge: Optional[list]) -> pd.DataFrame:
        groups = self._normalize_merge_groups(df, col_merge)
        if not groups:
            return df

        cols = list(df.columns)
        col_to_group = {}
        for gi, gcols in enumerate(groups):
            for c in gcols:
                if c not in col_to_group:
                    col_to_group[c] = gi

        group_first_col = {}
        for c in cols:
            gi = col_to_group.get(c, None)
            if gi is not None and gi not in group_first_col:
                group_first_col[gi] = c

        merged_series = {}
        new_cols = []
        for c in cols:
            gi = col_to_group.get(c, None)
            if gi is None:
                new_cols.append(c)
            else:
                if group_first_col.get(gi) == c:
                    gcols = groups[gi]
                    name = "+".join(str(x) for x in gcols)
                    s = df[gcols[0]].astype(str)
                    for gc in gcols[1:]:
                        s = s + df[gc].astype(str)
                    merged_series[name] = s
                    new_cols.append(name)
                else:
                    continue

        out = pd.DataFrame(index=df.index)
        for c in new_cols:
            if c in merged_series:
                out[c] = merged_series[c]
            else:
                out[c] = df[c]
        return out

    def _col_level_score_factory(self, sample_hashes: List[List[int]], sample_lens: List[List[int]]):
        m = len(sample_hashes)
        k = len(sample_hashes[0]) if m else 0
        mask = (1 << 64) - 1
        prime = 1099511628211
        offset = 1469598103934665603

        def score(order: Tuple[int, ...]) -> int:
            seen = [set() for _ in range(m)]
            ph = [0] * m
            pl = [0] * m
            total = 0
            sh = sample_hashes
            sl = sample_lens

            for r in range(k):
                h = offset
                plen = 0
                for t, c in enumerate(order):
                    h = ((h ^ sh[c][r]) * prime) & mask
                    plen += sl[c][r]
                    ph[t] = h
                    pl[t] = plen

                add_len = 0
                for t in range(m - 1, -1, -1):
                    if ph[t] in seen[t]:
                        add_len = pl[t]
                        break
                total += add_len

                for t in range(m):
                    seen[t].add(ph[t])

            return total

        return score

    def _trie_score(self, sample_cols: List[List[str]], order: Tuple[int, ...]) -> int:
        k = len(sample_cols[0]) if sample_cols else 0
        if k <= 1:
            return 0

        nodes = [dict()]
        total = 0
        cols = sample_cols

        for i in range(k):
            s = "".join(cols[c][i] for c in order)
            node = 0
            depth = 0
            matched = True
            for ch in s:
                d = nodes[node]
                nxt = d.get(ch)
                if nxt is None:
                    nxt = len(nodes)
                    nodes.append({})
                    d[ch] = nxt
                    matched = False
                else:
                    if matched:
                        depth += 1
                node = nxt
            if i > 0:
                total += depth
        return total

    def _compute_column_stats(self, df: pd.DataFrame):
        cols = list(df.columns)
        n = len(df)
        m = len(cols)

        nunique = [0] * m
        mean_len = [0.0] * m
        distinct_ratio = [0.0] * m
        repeat_len_mass = [0] * m

        col_str_full: List[np.ndarray] = []

        for i, c in enumerate(cols):
            arr_str = df[c].astype(str).to_numpy()
            col_str_full.append(arr_str)

            if n == 0:
                nunique[i] = 0
                mean_len[i] = 0.0
                distinct_ratio[i] = 0.0
                repeat_len_mass[i] = 0
                continue

            uniq = pd.unique(arr_str)
            nunique_i = int(len(uniq))
            nunique[i] = nunique_i
            distinct_ratio[i] = nunique_i / float(n) if n else 0.0

            total_len = 0
            for v in arr_str:
                total_len += len(v)
            mean_len[i] = total_len / float(n) if n else 0.0

            sum_unique_len = 0
            for u in uniq:
                sum_unique_len += len(u)
            repeat_len_mass[i] = int(total_len - sum_unique_len)

        return cols, col_str_full, nunique, mean_len, distinct_ratio, repeat_len_mass

    def _sample_indices(self, n: int, k: int) -> np.ndarray:
        if n <= 0:
            return np.zeros(0, dtype=np.int32)
        if k >= n:
            return np.arange(n, dtype=np.int32)
        if k <= 1:
            return np.array([0], dtype=np.int32)
        return np.linspace(0, n - 1, num=k, dtype=np.int32)

    def _build_sample_hash_len(self, col_str_full: List[np.ndarray], idx: np.ndarray):
        m = len(col_str_full)
        sample_hashes: List[List[int]] = []
        sample_lens: List[List[int]] = []
        mask = (1 << 64) - 1

        for ci in range(m):
            s = col_str_full[ci].take(idx, mode="clip")
            h = [(hash(x) & mask) for x in s.tolist()]
            l = [len(x) for x in s.tolist()]
            sample_hashes.append(h)
            sample_lens.append(l)

        return sample_hashes, sample_lens

    def _build_sample_strings(self, col_str_full: List[np.ndarray], idx: np.ndarray) -> List[List[str]]:
        cols_sample: List[List[str]] = []
        for arr in col_str_full:
            cols_sample.append(arr.take(idx, mode="clip").astype(str).tolist())
        return cols_sample

    def _unique_orders(self, orders: List[List[int]]) -> List[Tuple[int, ...]]:
        seen = set()
        out: List[Tuple[int, ...]] = []
        for o in orders:
            t = tuple(o)
            if t not in seen:
                seen.add(t)
                out.append(t)
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
        df2 = self._apply_col_merge(df, col_merge)
        cols, col_str_full, nunique, mean_len, distinct_ratio, repeat_len_mass = self._compute_column_stats(df2)

        n = len(df2)
        m = len(cols)
        if m <= 1 or n <= 1:
            return df2

        k1 = min(n, int(min(max(500, 0), early_stop)) if early_stop is not None else n)
        if k1 <= 0:
            k1 = min(n, 4000)
        k1 = min(n, max(1500, min(k1, 4500)))
        idx1 = self._sample_indices(n, k1)

        sample_hashes, sample_lens = self._build_sample_hash_len(col_str_full, idx1)
        col_score = self._col_level_score_factory(sample_hashes, sample_lens)

        base = list(range(m))
        orders: List[List[int]] = []

        orders.append(base[:])
        orders.append(base[::-1])

        orders.append(sorted(base, key=lambda i: (nunique[i], mean_len[i])))
        orders.append(sorted(base, key=lambda i: (distinct_ratio[i], mean_len[i])))
        orders.append(sorted(base, key=lambda i: (-repeat_len_mass[i], nunique[i], distinct_ratio[i])))
        orders.append(sorted(base, key=lambda i: (-mean_len[i] * (1.0 - distinct_ratio[i]), distinct_ratio[i])))
        orders.append(sorted(base, key=lambda i: (distinct_ratio[i], -repeat_len_mass[i], -mean_len[i])))

        low = [i for i in base if distinct_ratio[i] <= distinct_value_threshold]
        high = [i for i in base if distinct_ratio[i] > distinct_value_threshold]
        orders.append(sorted(low, key=lambda i: (-repeat_len_mass[i], nunique[i])) + sorted(high, key=lambda i: (-repeat_len_mass[i], nunique[i])))

        for o in list(orders):
            orders.append(o[::-1])

        rng = random.Random(0)
        seed_orders = self._unique_orders(orders)
        if seed_orders:
            for _ in range(18):
                o = list(seed_orders[0])
                rng.shuffle(o)
                orders.append(o)
        orders = [list(t) for t in self._unique_orders(orders)]

        scored = []
        for o in orders:
            s = col_score(tuple(o))
            scored.append((s, tuple(o)))
        scored.sort(reverse=True, key=lambda x: x[0])

        start_candidates = [t for _, t in scored[: min(6, len(scored))]]
        best_order = start_candidates[0] if start_candidates else tuple(base)
        best_score = col_score(best_order)

        def hillclimb(init_order: Tuple[int, ...], init_score: int) -> Tuple[Tuple[int, ...], int]:
            cur = list(init_order)
            cur_score = init_score
            improved = True
            it = 0
            while improved and it < 18:
                improved = False
                it += 1
                best_local_score = cur_score
                best_local = None
                for i in range(m - 1):
                    for j in range(i + 1, m):
                        cand = cur[:]
                        cand[i], cand[j] = cand[j], cand[i]
                        s = col_score(tuple(cand))
                        if s > best_local_score:
                            best_local_score = s
                            best_local = cand
                if best_local is not None:
                    cur = best_local
                    cur_score = best_local_score
                    improved = True
            return tuple(cur), cur_score

        for init in start_candidates:
            init_s = col_score(init)
            o2, s2 = hillclimb(init, init_s)
            if s2 > best_score:
                best_score = s2
                best_order = o2

        k2 = min(n, int(min(early_stop, 9000)) if early_stop is not None else n)
        k2 = min(n, max(2500, min(k2, 9000)))
        idx2 = self._sample_indices(n, k2)
        sample_cols2 = self._build_sample_strings(col_str_full, idx2)

        final_candidates = [best_order]
        for _, t in scored[: min(4, len(scored))]:
            final_candidates.append(t)
        final_candidates = list(dict.fromkeys(final_candidates))

        best_final = best_order
        best_final_score = self._trie_score(sample_cols2, best_order)

        for cand in final_candidates:
            sc = self._trie_score(sample_cols2, cand)
            if sc > best_final_score:
                best_final_score = sc
                best_final = cand

        final_cols = [cols[i] for i in best_final]
        return df2.loc[:, final_cols]