import numpy as np
import pandas as pd


class Solution:
    def _resolve_merge_group(self, df: pd.DataFrame, group):
        cols = list(df.columns)
        if not group:
            return []
        if all(isinstance(x, (int, np.integer)) for x in group):
            g = [int(x) for x in group]
            use_zero_based = any(x == 0 for x in g)
            resolved = []
            if use_zero_based:
                for x in g:
                    if 0 <= x < len(cols):
                        resolved.append(cols[x])
            else:
                for x in g:
                    if 1 <= x <= len(cols):
                        resolved.append(cols[x - 1])
            return [c for c in resolved if c in df.columns]
        else:
            resolved = []
            for x in group:
                if x in df.columns:
                    resolved.append(x)
                else:
                    xs = str(x)
                    if xs in df.columns:
                        resolved.append(xs)
            return resolved

    def _apply_col_merge(self, df: pd.DataFrame, col_merge: list):
        if not col_merge:
            return df

        df2 = df.copy()
        existing = set(df2.columns)
        used_new_names = set(existing)

        for gi, group in enumerate(col_merge):
            cols = self._resolve_merge_group(df2, group)
            cols = [c for c in cols if c in df2.columns]
            if len(cols) <= 1:
                continue

            base_name = "MERGE_" + "_".join(str(c) for c in cols)
            new_name = base_name
            if new_name in used_new_names:
                k = 1
                while f"{base_name}__{k}" in used_new_names:
                    k += 1
                new_name = f"{base_name}__{k}"
            used_new_names.add(new_name)

            arr = df2[cols[0]].astype(str).to_numpy()
            for c in cols[1:]:
                arr = np.char.add(arr, df2[c].astype(str).to_numpy())
            df2[new_name] = pd.Series(arr, index=df2.index)

            df2.drop(columns=cols, inplace=True, errors="ignore")

        return df2

    def _choose_sample_positions(self, n: int, k: int):
        if k >= n:
            return np.arange(n, dtype=np.int64)
        rng = np.random.default_rng(0)
        return np.sort(rng.choice(n, size=k, replace=False).astype(np.int64))

    def _compute_order(self, df: pd.DataFrame, sample_rows: int, distinct_value_threshold: float):
        cols = list(df.columns)
        m = len(cols)
        if m <= 1:
            return cols

        n = len(df)
        k = min(n, sample_rows)
        if k <= 1:
            return cols

        pos = self._choose_sample_positions(n, k)
        sdf = df.iloc[pos]

        codes_list = []
        bases = np.empty(m, dtype=np.int64)
        wlen = np.empty(m, dtype=np.float64)
        distinct_ratio = np.empty(m, dtype=np.float64)

        for idx, c in enumerate(cols):
            s = sdf[c].astype(str)
            codes, uniques = pd.factorize(s, sort=False)
            codes_list.append(codes.astype(np.int32, copy=False))
            ucnt = int(len(uniques))
            bases[idx] = ucnt + 2
            lmean = float(s.str.len().mean()) if k else 1.0
            wl = lmean
            if wl > 16.0:
                wl = 16.0
            elif wl < 1.0:
                wl = 1.0
            wlen[idx] = wl
            distinct_ratio[idx] = ucnt / float(k) if k else 1.0

        remaining = list(range(m))
        chosen = []
        group_id = None  # int32 array length k

        # Precompute initial scores using bincount (fast)
        initial_scores = np.empty(m, dtype=np.float64)
        for ci in range(m):
            codes = codes_list[ci]
            cs = (codes.astype(np.int64) + 1)
            cnt = np.bincount(cs, minlength=(int(cs.max()) + 1) if cs.size else 1)
            sumsq = float((cnt * cnt).sum())
            # slight penalty for extreme distinctness, mainly to avoid any length-weight artifacts
            pen = 1.0
            if distinct_ratio[ci] >= distinct_value_threshold:
                pen = 0.85
            initial_scores[ci] = sumsq * wlen[ci] * pen

        first = int(np.argmax(initial_scores))
        chosen.append(first)
        remaining.remove(first)
        group_id = (codes_list[first].astype(np.int32) + 1)

        # Greedy append
        gid = group_id
        k_int = int(k)
        for _ in range(m - 1):
            if not remaining:
                break

            best_c = None
            best_score = -1.0

            gid_local = gid.astype(np.int64, copy=False)
            for ci in remaining:
                base = int(bases[ci])
                codes = (codes_list[ci].astype(np.int64) + 1)
                counts = {}
                get = counts.get
                score = 0

                # sum of squares of counts in (group_id, code) bins
                # incremental update: when count t -> t+1, sumsq += 2t+1
                for i in range(k_int):
                    key = int(gid_local[i] * base + codes[i])
                    t = get(key, 0)
                    score += (t << 1) + 1
                    counts[key] = t + 1

                sc = float(score) * float(wlen[ci])
                if distinct_ratio[ci] >= distinct_value_threshold:
                    sc *= 0.92

                if sc > best_score:
                    best_score = sc
                    best_c = ci

            if best_c is None:
                break

            chosen.append(best_c)
            remaining.remove(best_c)

            base = int(bases[best_c])
            codes = (codes_list[best_c].astype(np.int64) + 1)
            comb = gid_local * base + codes
            gid, _ = pd.factorize(comb, sort=False)
            gid = (gid.astype(np.int32) + 1)

        ordered_cols = [cols[i] for i in chosen] + [cols[i] for i in remaining]
        return ordered_cols

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

        m = df2.shape[1]
        if m <= 1:
            return df2

        n = len(df2)
        # Keep sampling small and stable for runtime; early_stop used as an upper bound hint.
        sample_rows = 2000
        if isinstance(early_stop, int) and early_stop > 0:
            sample_rows = min(sample_rows, early_stop)
        sample_rows = max(400, min(sample_rows, n))

        ordered_cols = self._compute_order(df2, sample_rows=sample_rows, distinct_value_threshold=distinct_value_threshold)
        return df2.loc[:, ordered_cols]