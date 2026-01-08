import pandas as pd
import numpy as np
import random
from typing import List, Optional, Dict, Tuple


class Solution:
    def _apply_col_merge(self, df: pd.DataFrame, col_merge: Optional[list]) -> pd.DataFrame:
        if not col_merge:
            return df

        df2 = df
        used = set()

        for group in col_merge:
            if not group or len(group) < 2:
                continue

            cols = [c for c in group if c in df2.columns and c not in used]
            if len(cols) < 2:
                continue

            new_name = "_".join(cols)
            merged = df2[cols[0]].astype(str)
            for c in cols[1:]:
                merged = merged + df2[c].astype(str)

            old_cols = list(df2.columns)
            insert_at = old_cols.index(cols[0])

            df2 = df2.drop(columns=cols)
            df2[new_name] = merged.values

            new_cols = list(df2.columns)
            new_cols.remove(new_name)
            new_cols.insert(insert_at, new_name)
            df2 = df2[new_cols]

            used.update(cols)

        return df2

    @staticmethod
    def _col_prefix_potential(strings: np.ndarray, max_L: int = 4) -> int:
        best = 0
        for L in range(1, max_L + 1):
            d = {}
            for s in strings:
                p = s[:L]
                d[p] = d.get(p, 0) + 1
            pot = 0
            for c in d.values():
                if c > 1:
                    pot += (c - 1) * L
            if pot > best:
                best = pot
        return best

    @staticmethod
    def _eval_perm(
        perm: Tuple[int, ...],
        codes_cols: List[np.ndarray],
        lens_cols: List[np.ndarray],
        n_rows: int,
        base: int = 1000003,
    ) -> int:
        m = len(perm)
        seen = [set() for _ in range(m)]
        mask = (1 << 64) - 1

        total = 0
        hs = [0] * m

        for i in range(n_rows):
            h = 0
            matched = 0
            checking = True
            for k in range(m):
                c = perm[k]
                h = (h * base + int(codes_cols[c][i]) + 1) & mask
                hs[k] = h
                if checking:
                    if h in seen[k]:
                        matched += int(lens_cols[c][i])
                    else:
                        checking = False
            if i:
                total += matched
            for k in range(m):
                seen[k].add(hs[k])

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
        df2 = self._apply_col_merge(df, col_merge)
        cols = list(df2.columns)
        m = len(cols)
        if m <= 1:
            return df2

        n = len(df2)
        if n <= 1:
            return df2

        codes_cols: List[np.ndarray] = []
        lens_cols: List[np.ndarray] = []

        distinct_ratio = np.zeros(m, dtype=np.float64)
        avg_len = np.zeros(m, dtype=np.float64)
        matchable_ratio = np.zeros(m, dtype=np.float64)
        early_score = np.zeros(m, dtype=np.float64)

        for ci, c in enumerate(cols):
            s = df2[c].astype(str).to_numpy(dtype=object, copy=False)
            codes, uniques = pd.factorize(s, sort=False)
            codes_cols.append(codes.astype(np.uint32, copy=False))

            lens = np.fromiter((len(x) for x in s), dtype=np.int32, count=n)
            lens_cols.append(lens)

            nunique = len(uniques)
            distinct_ratio[ci] = nunique / float(n)

            total_len = int(lens.sum())
            avg_len[ci] = total_len / float(n)

            unique_len_sum = sum(len(u) for u in uniques.tolist())
            matchable = total_len - unique_len_sum
            matchable_ratio[ci] = matchable / float(total_len + 1e-9)

            prefix_pot = self._col_prefix_potential(s, max_L=4)
            early_score[ci] = float(matchable) + 0.15 * float(prefix_pot)

        def rest_order(remaining: List[int]) -> List[int]:
            remaining.sort(
                key=lambda x: (
                    distinct_ratio[x] >= distinct_value_threshold,
                    distinct_ratio[x],
                    -matchable_ratio[x],
                    -avg_len[x],
                )
            )
            return remaining

        # sample size: 10**row_stop (default 10000), clipped
        sample_n = min(n, max(2000, min(12000, 10 ** max(1, int(row_stop)))))
        # deterministic randomness per dataset schema
        seed = 146959810
        for c in cols:
            seed = (seed * 16777619) ^ (hash(c) & 0xFFFFFFFF)
        rng = random.Random(seed)

        eval_cache_sample: Dict[Tuple[int, ...], int] = {}
        eval_cache_full: Dict[Tuple[int, ...], int] = {}

        def eval_sample(perm: Tuple[int, ...]) -> int:
            v = eval_cache_sample.get(perm)
            if v is None:
                v = self._eval_perm(perm, codes_cols, lens_cols, sample_n)
                eval_cache_sample[perm] = v
            return v

        def eval_full(perm: Tuple[int, ...]) -> int:
            v = eval_cache_full.get(perm)
            if v is None:
                v = self._eval_perm(perm, codes_cols, lens_cols, n)
                eval_cache_full[perm] = v
            return v

        all_idx = list(range(m))

        # Candidate pool
        candidates = []

        # original order
        candidates.append(tuple(all_idx))

        # heuristics
        candidates.append(tuple(sorted(all_idx, key=lambda x: distinct_ratio[x])))
        candidates.append(tuple(sorted(all_idx, key=lambda x: -matchable_ratio[x])))
        candidates.append(tuple(sorted(all_idx, key=lambda x: -early_score[x])))

        # beam search with heuristic fill
        beam_width = min(30, max(4, 2 ** (max(1, int(col_stop)) + 1)))
        beam: List[Tuple[List[int], int]] = [([], -1)]
        eval_count = 0

        for depth in range(m):
            next_beam: List[Tuple[List[int], int]] = []
            for partial, _ in beam:
                used = set(partial)
                rem = [x for x in all_idx if x not in used]
                if not rem:
                    continue
                # prioritize trying better columns earlier
                rem_sorted = sorted(
                    rem,
                    key=lambda x: (
                        distinct_ratio[x] >= distinct_value_threshold,
                        distinct_ratio[x],
                        -early_score[x],
                    ),
                )
                for c in rem_sorted:
                    if eval_count >= early_stop:
                        break
                    new_partial = partial + [c]
                    rem2 = [x for x in all_idx if x not in set(new_partial)]
                    rem2 = rest_order(rem2)
                    full_perm = tuple(new_partial + rem2)
                    score = eval_sample(full_perm)
                    eval_count += 1
                    next_beam.append((new_partial, score))
                if eval_count >= early_stop:
                    break
            if not next_beam:
                break
            next_beam.sort(key=lambda x: x[1], reverse=True)
            beam = next_beam[:beam_width]

        for partial, _ in beam:
            used = set(partial)
            rem = [x for x in all_idx if x not in used]
            rem = rest_order(rem)
            candidates.append(tuple(partial + rem))

        # random guided candidates
        base_order = list(sorted(all_idx, key=lambda x: (distinct_ratio[x] >= distinct_value_threshold, distinct_ratio[x], -early_score[x])))
        for _ in range(min(12, early_stop // 50 + 2)):
            tmp = base_order[:]
            # shuffle within blocks (low distinct then high distinct) to keep structure
            split = 0
            for i in range(m):
                if distinct_ratio[tmp[i]] >= distinct_value_threshold:
                    split = i
                    break
            else:
                split = m
            low = tmp[:split]
            high = tmp[split:]
            rng.shuffle(low)
            rng.shuffle(high)
            candidates.append(tuple(low + high))

        # unique candidates
        seen_cand = set()
        uniq_candidates = []
        for p in candidates:
            if p not in seen_cand:
                seen_cand.add(p)
                uniq_candidates.append(p)

        # choose best on sample
        scored = [(eval_sample(p), p) for p in uniq_candidates]
        scored.sort(key=lambda x: x[0], reverse=True)
        best_sample_perm = scored[0][1]

        # sample-based local swap refinement
        best_perm = list(best_sample_perm)
        best_score = eval_sample(tuple(best_perm))
        for _ in range(2):
            improved = False
            for i in range(m - 1):
                for j in range(i + 1, m):
                    cand = best_perm[:]
                    cand[i], cand[j] = cand[j], cand[i]
                    s = eval_sample(tuple(cand))
                    if s > best_score:
                        best_score = s
                        best_perm = cand
                        improved = True
            if not improved:
                break

        # Evaluate top-K on full
        topK = min(6, len(scored))
        final_pool = [best_perm]
        for _, p in scored[:topK]:
            final_pool.append(list(p))

        # add a few swap-neighborhood candidates from refined best
        refined = best_perm
        for _ in range(min(10, m * (m - 1) // 2)):
            i = rng.randrange(m)
            j = rng.randrange(m)
            if i == j:
                continue
            cand = refined[:]
            cand[i], cand[j] = cand[j], cand[i]
            final_pool.append(cand)

        seen_final = set()
        final_uniq = []
        for p in final_pool:
            t = tuple(p)
            if t not in seen_final:
                seen_final.add(t)
                final_uniq.append(t)

        best_full = final_uniq[0]
        best_full_score = eval_full(best_full)
        for p in final_uniq[1:]:
            s = eval_full(p)
            if s > best_full_score:
                best_full_score = s
                best_full = p

        out_cols = [cols[i] for i in best_full]
        return df2[out_cols]