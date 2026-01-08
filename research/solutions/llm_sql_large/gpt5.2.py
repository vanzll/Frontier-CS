import pandas as pd
import numpy as np


class Solution:
    def _unique_col_name(self, existing, base):
        if base not in existing:
            return base
        i = 1
        while True:
            name = f"{base}__{i}"
            if name not in existing:
                return name
            i += 1

    def _normalize_merge_group(self, df_cols, group):
        if not group:
            return []
        names = []
        for x in group:
            if isinstance(x, (int, np.integer)):
                ix = int(x)
                if 0 <= ix < len(df_cols):
                    names.append(df_cols[ix])
            else:
                names.append(x)
        # preserve order; drop non-existing; de-dup
        seen = set()
        out = []
        for n in names:
            if n in df_cols and n not in seen:
                out.append(n)
                seen.add(n)
        return out

    def _apply_col_merge(self, df: pd.DataFrame, col_merge):
        if not col_merge:
            return df
        df = df.copy()
        for grp in col_merge:
            cols = list(df.columns)
            names = self._normalize_merge_group(cols, grp)
            if len(names) <= 1:
                continue

            pos = min(df.columns.get_loc(c) for c in names if c in df.columns)
            merged = df[names[0]].astype(str)
            for c in names[1:]:
                merged = merged + df[c].astype(str)

            base_name = "__merge__" + "__".join(str(x) for x in names)
            new_name = self._unique_col_name(set(df.columns), base_name)

            df = df.drop(columns=names)
            df.insert(pos, new_name, merged)
        return df

    def _sample_indices(self, n, sample_size):
        if n <= sample_size:
            return np.arange(n, dtype=np.int64)
        head_n = min(1000, max(0, sample_size // 2))
        remaining = sample_size - head_n
        if remaining <= 0:
            return np.arange(head_n, dtype=np.int64)
        rng = np.random.default_rng(0)
        rand_idx = rng.choice(np.arange(head_n, n, dtype=np.int64), size=remaining, replace=False)
        return np.concatenate([np.arange(head_n, dtype=np.int64), rand_idx])

    def _col_stats_and_codes(self, sample_series: pd.Series):
        s = sample_series.astype(str).to_numpy(dtype=object, copy=False)
        n = s.shape[0]
        if n == 0:
            codes = np.zeros((0,), dtype=np.int32)
            return 0.0, 0.0, 1.0, (1.0, 1.0, 1.0), 0.0, codes

        lengths = np.fromiter((len(x) for x in s), dtype=np.int32, count=n)
        avg_len = float(lengths.mean()) if n else 0.0

        codes, _ = pd.factorize(s, sort=False)
        codes = codes.astype(np.int32, copy=False)

        if n >= 2:
            bc = np.bincount(codes)
            num = float(np.sum(bc * (bc - 1)))
            denom = float(n * (n - 1))
            match_prob = num / denom if denom > 0 else 0.0
        else:
            match_prob = 0.0

        distinct_ratio = float(np.unique(codes).size) / float(n) if n else 1.0

        # prefix concentration for k=1..3 (max frequency)
        max_freqs = []
        for k in (1, 2, 3):
            d = {}
            for x in s:
                p = x[:k]
                d[p] = d.get(p, 0) + 1
            max_freqs.append(max(d.values()) / float(n) if d else 1.0)
        max1, max2, max3 = max_freqs

        prefix_weight = 1.0 * max1 + 2.0 * max2 + 3.0 * max3
        base_score = avg_len * (match_prob ** 0.5) + 0.3 * prefix_weight

        return avg_len, match_prob, distinct_ratio, (max1, max2, max3), base_score, codes

    def _update_partition(self, partition, codes_col):
        new_part = []
        for grp in partition:
            if grp.size <= 1:
                new_part.append(grp)
                continue
            arr = codes_col[grp]
            order = np.argsort(arr, kind="mergesort")
            sorted_idx = grp[order]
            sorted_arr = arr[order]
            if sorted_arr.size <= 1:
                new_part.append(sorted_idx)
                continue
            cuts = np.flatnonzero(sorted_arr[1:] != sorted_arr[:-1]) + 1
            if cuts.size == 0:
                new_part.append(sorted_idx)
            else:
                splits = np.split(sorted_idx, cuts)
                new_part.extend(splits)
        return new_part

    def _greedy_prefix_columns(self, cols, codes_by_col, avg_lens, base_scores, steps=8, max_groups=500, eval_top_groups=80):
        m = len(cols)
        if m == 0:
            return []

        chosen = []
        remaining = list(range(m))
        partition = [np.arange(codes_by_col[0].shape[0], dtype=np.int32)]

        for _ in range(min(steps, m)):
            if not remaining:
                break
            if len(partition) > max_groups:
                break

            part_sorted = sorted(partition, key=lambda a: a.size, reverse=True)
            part_eval = part_sorted[:eval_top_groups]

            best_idx = None
            best_score = -1e30

            for ci in remaining:
                avg_len = avg_lens[ci]
                if avg_len <= 0:
                    cand_score = 0.01 * base_scores[ci]
                else:
                    acc = 0.0
                    for grp in part_eval:
                        g = int(grp.size)
                        if g <= 1:
                            continue
                        arr = codes_by_col[ci][grp]
                        _, counts = np.unique(arr, return_counts=True)
                        num = float(np.sum(counts * (counts - 1)))
                        denom = float(g * (g - 1))
                        p = (num / denom) if denom > 0 else 0.0
                        acc += (g / float(codes_by_col[ci].shape[0])) * (p ** 0.5)
                    cand_score = avg_len * acc + 0.05 * base_scores[ci]

                if cand_score > best_score:
                    best_score = cand_score
                    best_idx = ci

            if best_idx is None:
                break
            chosen.append(best_idx)
            remaining.remove(best_idx)
            partition = self._update_partition(partition, codes_by_col[best_idx])

            # stop if partition becomes too fine
            if len(partition) > max_groups:
                break

        return [cols[i] for i in chosen], [i for i in remaining]

    def _enforce_one_way_deps(self, order, one_way_dep, df_cols):
        if not one_way_dep:
            return order
        pos = {c: i for i, c in enumerate(order)}
        deps = []
        for a, b in one_way_dep:
            if isinstance(a, (int, np.integer)):
                ia = int(a)
                if 0 <= ia < len(df_cols):
                    a = df_cols[ia]
            if isinstance(b, (int, np.integer)):
                ib = int(b)
                if 0 <= ib < len(df_cols):
                    b = df_cols[ib]
            if a in pos and b in pos and a != b:
                deps.append((a, b))
        if not deps:
            return order

        order = list(order)
        for _ in range(min(5 * len(deps), 200)):
            changed = False
            pos = {c: i for i, c in enumerate(order)}
            for a, b in deps:
                ia, ib = pos.get(a), pos.get(b)
                if ia is None or ib is None:
                    continue
                if ia > ib:
                    order.pop(ia)
                    ib = pos[b]  # old position of b, may have shifted by pop if ia<ib (not here)
                    pos = {c: i for i, c in enumerate(order)}
                    ib = pos[b]
                    order.insert(ib, a)
                    changed = True
                    break
            if not changed:
                break
        return order

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
        n = len(df2)
        m = len(cols)
        if m <= 1 or n == 0:
            return df2

        sample_size = int(min(n, 4000))
        if early_stop is not None and early_stop > 0:
            sample_size = min(sample_size, int(early_stop), n)
        if sample_size <= 0:
            return df2

        idx = self._sample_indices(n, sample_size)
        sample = df2.iloc[idx]

        avg_lens = np.zeros(m, dtype=np.float32)
        match_probs = np.zeros(m, dtype=np.float32)
        distinct_ratios = np.zeros(m, dtype=np.float32)
        base_scores = np.zeros(m, dtype=np.float32)
        prefix1 = np.zeros(m, dtype=np.float32)
        codes_by_col = []

        for i, c in enumerate(cols):
            a, mp, dr, (mx1, mx2, mx3), bs, codes = self._col_stats_and_codes(sample[c])
            avg_lens[i] = a
            match_probs[i] = mp
            distinct_ratios[i] = dr
            base_scores[i] = bs
            prefix1[i] = mx1
            codes_by_col.append(codes)

        chosen_cols, remaining_idx = self._greedy_prefix_columns(
            cols,
            codes_by_col,
            avg_lens,
            base_scores,
            steps=min(8, m),
            max_groups=500,
            eval_top_groups=80,
        )

        chosen_set = set(chosen_cols)
        remaining_cols = [cols[i] for i in remaining_idx if cols[i] not in chosen_set]

        # push very high-distinct, low-prefix columns to the end
        low = []
        high = []
        col_to_i = {c: i for i, c in enumerate(cols)}
        for c in remaining_cols:
            i = col_to_i[c]
            dr = float(distinct_ratios[i])
            mp = float(match_probs[i])
            mx1 = float(prefix1[i])
            if dr > distinct_value_threshold and mp < 0.02 and mx1 < 0.35:
                high.append(c)
            else:
                low.append(c)

        low.sort(key=lambda c: float(base_scores[col_to_i[c]]), reverse=True)
        high.sort(key=lambda c: float(base_scores[col_to_i[c]]), reverse=True)

        final_order = chosen_cols + low + high
        if one_way_dep:
            final_order = self._enforce_one_way_deps(final_order, one_way_dep, list(df2.columns))

        # ensure all columns included exactly once
        seen = set()
        dedup = []
        for c in final_order:
            if c in df2.columns and c not in seen:
                dedup.append(c)
                seen.add(c)
        for c in df2.columns:
            if c not in seen:
                dedup.append(c)

        return df2[dedup]