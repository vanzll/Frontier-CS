import pandas as pd
import numpy as np


class Solution:
    def _apply_merges(self, df: pd.DataFrame, col_merge):
        if not col_merge:
            return df

        df = df.copy()
        used_cols = set()
        new_cols = []
        merge_count = 0

        # Helper to resolve column names from strings/ints
        def resolve_name(x):
            if isinstance(x, int):
                cols = list(df.columns)
                if x < 0:
                    idx = len(cols) + x
                else:
                    idx = x
                if idx < 0 or idx >= len(cols):
                    return None
                return cols[idx]
            elif isinstance(x, str):
                return x if x in df.columns else None
            else:
                return None

        for grp in col_merge:
            # Resolve actual column names, skip ones not in df
            names = []
            for g in grp:
                nm = resolve_name(g)
                if nm is not None and nm in df.columns and nm not in used_cols:
                    names.append(nm)
            if not names:
                continue
            if len(names) == 1:
                # Single column "merge": just keep as is but we will still drop and create a new one to avoid duplicates
                arr = df[names[0]].astype(str).to_numpy(copy=False)
                new_name = f"MERGE_{merge_count}"
                df[new_name] = arr
                used_cols.update(names)
                merge_count += 1
                new_cols.append(new_name)
                continue

            # Concatenate strings of columns in names
            arrays = [df[name].astype(str).to_numpy(copy=False) for name in names]
            res = arrays[0].copy()
            for k in range(1, len(arrays)):
                res = res + arrays[k]
            new_name = f"MERGE_{merge_count}"
            df[new_name] = res
            used_cols.update(names)
            merge_count += 1
            new_cols.append(new_name)

        # Drop used original columns
        if used_cols:
            df = df.drop(columns=list(used_cols))

        # Ensure merged columns appear just after remaining ones (we'll reorder later anyway)
        # Keep current df column order with new MERGE_* appended (already appended)

        return df

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
        # 1) Apply column merges
        work_df = self._apply_merges(df, col_merge)

        # 2) Precompute per-column string arrays, codes, lengths, stats
        col_names = list(work_df.columns)
        M = len(col_names)
        N = len(work_df)

        if M <= 1:
            return work_df

        # Sampling indices for evaluation to speed up
        nlimit = min(N, max(1, early_stop))
        step = max(1, int(row_stop))
        sample_idx = np.arange(0, nlimit, step, dtype=np.int64)

        # Precompute strings, lengths, factorize codes
        codes_list = []
        codes_u64_list = []
        lens_list = []
        uniq_ratio = []
        avg_len = []
        sum_len = []
        base_score = []

        # Precompute salts per column for hashing
        # Use stable salts derived from column name string
        def stable_salt(name):
            # FNV-1a like hash to 64-bit
            h = np.uint64(1469598103934665603)
            for ch in name:
                h = h ^ np.uint64(ord(ch))
                h = h * np.uint64(1099511628211)
            # ensure non-zero, odd
            val = int(h & np.uint64(0xFFFFFFFFFFFFFFFF))
            if val == 0:
                val = 0x9E3779B97F4A7C15
            if val % 2 == 0:
                val ^= 1
            return np.uint64(val)

        salts = [stable_salt(nm) for nm in col_names]

        for c in col_names:
            # Convert to strings and factorize
            s = work_df[c].astype(str)
            arr = s.to_numpy(copy=False)
            # lengths
            lens = np.fromiter((len(x) for x in arr), dtype=np.int32, count=arr.shape[0])
            # factorize
            codes, uniques = pd.factorize(arr, sort=False, na_sentinel=-1)
            # There should be no -1 if astype(str), but guard anyway
            if np.any(codes < 0):
                # Replace -1 with new code at end
                min_code = codes.min()
                if min_code < 0:
                    # map -1 to max+1
                    max_code = codes.max()
                    codes = codes.copy()
                    codes[codes < 0] = max_code + 1
                    uniques = np.append(uniques, "<NA>")
            codes_list.append(codes)
            codes_u64_list.append(codes.astype(np.uint64, copy=False))
            lens_list.append(lens)
            ucount = int(np.max(codes)) + 1 if codes.size > 0 else 0
            if ucount <= 0:
                ur = 1.0
            else:
                ur = ucount / float(N if N > 0 else 1)
            uniq_ratio.append(ur)
            sl = float(lens.sum())
            sum_len.append(sl)
            avg_len.append(float(lens.mean() if lens.size > 0 else 0.0))
            # base_score: sum of lengths for rows that are NOT the first occurrence of their value
            if ucount <= 1:
                # all same value
                # All except first occurrence contribute
                base_score_val = sl - float(lens[0]) if lens.size > 0 else 0.0
            else:
                visited = np.zeros(ucount, dtype=np.int8)
                acc = 0
                for i in range(N):
                    code = codes[i]
                    if visited[code]:
                        acc += int(lens[i])
                    else:
                        visited[code] = 1
                base_score_val = float(acc)
            base_score.append(base_score_val)

        uniq_ratio = np.array(uniq_ratio, dtype=float)
        avg_len = np.array(avg_len, dtype=float)
        sum_len = np.array(sum_len, dtype=float)
        base_score = np.array(base_score, dtype=float)

        # Priority metric to choose which columns to evaluate precisely
        # Demote columns with very high uniqueness
        priority_metric = base_score * (1.0 + avg_len * 0.0)  # base_score is already length-weighted
        high_unique_mask = uniq_ratio > float(distinct_value_threshold)
        priority_metric = priority_metric * np.where(high_unique_mask, 0.2, 1.0)

        # Prepare hashing of prefixes for all rows: start with zeros
        g_all = np.zeros(N, dtype=np.uint64)

        remaining = list(range(M))
        selected = []

        # evaluation-of-candidate function using sample
        FNV_PRIME = np.uint64(1099511628211)
        MIX_A = np.uint64(1315423911)

        # Helper to evaluate candidate incremental gain
        def eval_candidate(idx):
            # Evaluate incremental matched length using sample rows
            codes = codes_u64_list[idx]
            lens = lens_list[idx]
            salt = salts[idx]
            seen = set()
            # Compute on sample rows in original order
            total = 0
            g_samp = g_all  # alias
            for i in sample_idx:
                key = int((g_samp[i] * MIX_A) ^ (codes[i] * salt))
                if key in seen:
                    total += int(lens[i])
                else:
                    seen.add(key)
            return total

        # Iteratively build order
        while remaining:
            # number of candidates to evaluate precisely
            k = max(1, int(col_stop))
            k = min(k, len(remaining))

            # Pick top-k by priority metric among remaining
            rem_metrics = [(priority_metric[i], i) for i in remaining]
            # Partial selection: nlargest k
            # Use simple sort for small M (<= 68)
            rem_metrics.sort(key=lambda x: x[0], reverse=True)
            cand_indices = [i for (_, i) in rem_metrics[:k]]

            # Ensure that if all priorities are zero (e.g., extremes), we still select some candidates
            if not cand_indices:
                cand_indices = remaining[:k]

            # Evaluate candidates precisely
            best_gain = -1
            best_idx = cand_indices[0]
            for idx in cand_indices:
                gain = eval_candidate(idx)
                if gain > best_gain:
                    best_gain = gain
                    best_idx = idx

            # Append best to selected
            selected.append(best_idx)
            remaining.remove(best_idx)

            # Update prefix hashes for all rows
            codes_u = codes_u64_list[best_idx]
            salt = salts[best_idx]
            # g = (g * A) ^ (codes * salt)
            g_all = (g_all * MIX_A) ^ (codes_u * salt)
            g_all = g_all * FNV_PRIME  # spread further

        # Reorder columns accordingly
        ordered_cols = [col_names[i] for i in selected]
        result_df = work_df[ordered_cols]
        return result_df