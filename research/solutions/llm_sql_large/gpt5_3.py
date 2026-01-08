import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Tuple

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
        # Work on a shallow copy to avoid modifying the input df
        df_work = df.copy(deep=False)

        # 1) Apply column merges if specified
        if col_merge:
            existing_cols = set(df_work.columns)
            used_cols = set()
            merge_idx = 0
            for group in col_merge:
                if not group:
                    continue
                cols = [c for c in group if c in existing_cols]
                cols = [c for c in cols if c not in used_cols]
                if len(cols) <= 1:
                    continue
                # Create a unique merged column name
                base_name = "|".join(cols)
                new_name = f"MERGED_{merge_idx}"
                merge_idx += 1
                # Concatenate as strings without separator, consistent with problem statement
                # Using vectorized str.cat iteratively
                s = df_work[cols[0]].astype(str)
                for c in cols[1:]:
                    s = s.str.cat(df_work[c].astype(str), na_rep='')
                df_work[new_name] = s
                used_cols.update(cols)
            if used_cols:
                df_work = df_work[[c for c in df_work.columns if c not in used_cols] + [c for c in df_work.columns if c.startswith("MERGED_")]]

        cols = list(df_work.columns)
        N = len(df_work)
        if N == 0 or len(cols) <= 1:
            return df_work

        # Precompute codes, lengths, distinct ratios, and single-column duplicate contributions
        col_codes: Dict[str, np.ndarray] = {}
        col_lens: Dict[str, np.ndarray] = {}
        dup_contrib1: Dict[str, float] = {}
        distinct_ratio: Dict[str, float] = {}
        sum_len: Dict[str, int] = {}

        for c in cols:
            s = df_work[c].astype(str)
            lengths = s.str.len().to_numpy(dtype=np.int32, copy=False)
            codes, uniques = pd.factorize(s, sort=False)
            codes = codes.astype(np.int64, copy=False)
            dup_mask = s.duplicated(keep='first').to_numpy(dtype=bool, copy=False)
            dup_contr = float(lengths[dup_mask].sum())

            col_codes[c] = codes
            col_lens[c] = lengths
            dup_contrib1[c] = dup_contr
            distinct_ratio[c] = (len(uniques) / max(1, N))
            sum_len[c] = int(lengths.sum())

        # Partition columns by distinct ratio
        low_card_cols = [c for c in cols if distinct_ratio[c] <= distinct_value_threshold]
        high_card_cols = [c for c in cols if c not in low_card_cols]

        # Seeds for hashing per column
        rng = np.random.default_rng(123456789)
        col_seed: Dict[str, np.uint64] = {c: np.uint64(rng.integers(1, np.iinfo(np.int64).max)) for c in cols}

        # Hash combine function
        P1 = np.uint64(11400714819323198485)  # 2^64 / golden ratio
        P2 = np.uint64(14029467366897019727)  # splitmix64 prime
        P3 = np.uint64(1099511628211)         # FNV prime

        def init_hash(codes: np.ndarray, seed: np.uint64) -> np.ndarray:
            # Initialize hash from single column codes
            # (codes * P2) ^ seed
            return (np.asarray(codes, dtype=np.uint64) * P2) ^ seed

        def mix_hash(prev: np.ndarray, codes: np.ndarray, seed: np.uint64) -> np.ndarray:
            # Combine previous hash with new column codes
            # prev = (prev * P1) ^ ((codes * P3) + seed)
            return (prev * P1) ^ ((np.asarray(codes, dtype=np.uint64) * P3) + seed)

        # Greedy selection among low-cardinality columns using exact contribution
        selected: List[str] = []
        if low_card_cols:
            # First column: choose by single-column duplicate contribution
            first_col = max(low_card_cols, key=lambda c: (dup_contrib1[c], sum_len[c], -distinct_ratio[c]))
            selected.append(first_col)
            prefix_hash = init_hash(col_codes[first_col], col_seed[first_col])

            # Determine max greedy steps based on dataset size and parameters
            M = len(cols)
            # A conservative cap to keep runtime low
            base_steps = max(4, min(12, (row_stop if row_stop else 4) + (col_stop if col_stop else 2) + 2))
            if N * M > 2_500_000:
                base_steps = min(base_steps, 8)
            if N * M > 4_500_000:
                base_steps = min(base_steps, 6)
            max_greedy_steps = int(min(len(low_card_cols), base_steps))

            # Subsequent columns: greedy by conditional duplicate contribution
            while len(selected) < max_greedy_steps:
                candidates = [c for c in low_card_cols if c not in selected]
                if not candidates:
                    break
                best_c = None
                best_contrib = -1.0

                for c in candidates:
                    codes_c = col_codes[c]
                    lens_c = col_lens[c]
                    h_new = mix_hash(prefix_hash, codes_c, col_seed[c])
                    # Unique and contributions: sums per key except first occurrence per key
                    uniq, inverse, counts, first_idx = np.unique(
                        h_new, return_inverse=True, return_counts=True, return_index=True
                    )
                    sums = np.bincount(inverse, weights=lens_c).astype(np.float64, copy=False)
                    mask = counts >= 2
                    if mask.any():
                        contrib = float(sums[mask].sum() - lens_c[first_idx[mask]].sum())
                    else:
                        contrib = 0.0

                    if contrib > best_contrib:
                        best_contrib = contrib
                        best_c = c

                if best_c is None or best_contrib <= 0.0:
                    break

                selected.append(best_c)
                prefix_hash = mix_hash(prefix_hash, col_codes[best_c], col_seed[best_c])

        # Order remaining columns
        remaining_low = [c for c in low_card_cols if c not in selected]
        # Sort remaining low-card by single duplicate contribution desc, then by lower distinct ratio, then by longer total lengths
        remaining_low.sort(key=lambda c: (dup_contrib1[c], -distinct_ratio[c], sum_len[c]), reverse=True)

        # Order high-card columns: little impact on prefix overlap; choose by single dup contribution (likely zero), tie-breaker shorter mean length first
        def mean_len(c: str) -> float:
            return float(sum_len[c]) / max(1, N)

        high_card_cols_sorted = sorted(high_card_cols, key=lambda c: (dup_contrib1[c], -mean_len(c)))

        final_order = selected + remaining_low + high_card_cols_sorted

        # Ensure all columns are included and no duplicates
        seen = set()
        ordered_unique = []
        for c in final_order:
            if c not in seen and c in df_work.columns:
                seen.add(c)
                ordered_unique.append(c)
        # Add any leftover columns not accounted for (safety)
        for c in df_work.columns:
            if c not in seen:
                ordered_unique.append(c)

        return df_work[ordered_unique]