import pandas as pd
import numpy as np
from typing import List, Dict, Any


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
        # Step 1: Apply column merges if specified
        data = df.copy()
        if col_merge:
            # Ensure groups are lists of strings and exist in df
            used_cols = set()
            new_cols = {}
            for group in col_merge:
                if not group:
                    continue
                grp = [c for c in group if c in data.columns]
                if not grp:
                    continue
                # Create a unique name for merged column
                base_name = "MERGED(" + " + ".join(grp) + ")"
                name = base_name
                idx_suffix = 1
                while name in data.columns or name in new_cols:
                    name = f"{base_name}#{idx_suffix}"
                    idx_suffix += 1
                # Concatenate without separators to match scoring metric
                s = data[grp[0]].astype(str)
                for col in grp[1:]:
                    s = s + data[col].astype(str)
                new_cols[name] = s
                used_cols.update(grp)
            # Drop used columns and insert new merged columns at the end
            if used_cols:
                remaining = [c for c in data.columns if c not in used_cols]
                data = data[remaining]
            for name, s in new_cols.items():
                data[name] = s

        # Step 2: Prepare sample rows for scoring (early stopping on rows)
        n_rows_total = len(data)
        n_rows = min(early_stop if early_stop and early_stop > 0 else n_rows_total, n_rows_total)
        sample_df = data.iloc[:n_rows, :]

        columns = list(sample_df.columns)
        M = len(columns)
        if M <= 1:
            return data  # Nothing to reorder

        # Step 3: Precompute per-column codes, average lengths, and base stats
        col_codes: Dict[str, np.ndarray] = {}
        col_avg_len: Dict[str, float] = {}
        col_dup_frac: Dict[str, float] = {}
        col_run_bonus: Dict[str, float] = {}
        col_base_score: Dict[str, float] = {}

        # Hyperparameters for base scoring
        alpha = 1.8  # weight on duplication fraction
        beta = 1.0   # weight on average length
        gamma = 0.7  # weight on contiguity/run bonus

        for col in columns:
            s = sample_df[col].astype(str)
            # Factorize to integer codes; shift by +1 so NA -> 0, others >=1
            codes_raw, _ = pd.factorize(s, sort=False)
            codes = (codes_raw + 1).astype(np.uint32)  # now NA (if any) = 0, non-NA >=1
            col_codes[col] = codes

            # Unique count including NA as category if present
            has_na = np.any(codes == 0)
            unique_non_na = int(codes.max())  # since non-NA codes are 1..num_uniques
            unique_total = unique_non_na + (1 if has_na else 0)

            # Duplication fraction (1 - distinct_ratio)
            dup_frac = max(0.0, 1.0 - (unique_total / float(n_rows)))
            col_dup_frac[col] = dup_frac

            # Average string length
            avg_len = float(s.str.len().mean())
            if not np.isfinite(avg_len) or avg_len <= 0:
                avg_len = 0.0
            col_avg_len[col] = avg_len

            # Contiguity/run bonus based on consecutive equal codes
            if n_rows > 1:
                changes = np.count_nonzero(codes[1:] != codes[:-1])
                boundary_rate = changes / float(n_rows - 1)
                run_bonus = max(0.0, 1.0 - boundary_rate)  # higher is better
            else:
                run_bonus = 0.0
            col_run_bonus[col] = run_bonus

            # Base score for prioritization
            # Penalize highly distinct columns using alpha > 1
            base_score = (dup_frac ** alpha) * ((avg_len + 1e-12) ** beta) * ((run_bonus + 1e-6) ** gamma)
            col_base_score[col] = base_score

        # Step 4: Greedy selection using approximate expected incremental LCP
        # Maintain prefix group IDs to compute duplication fraction for extended prefixes quickly
        gid = np.zeros(n_rows, dtype=np.uint32)

        selected: List[str] = []
        remaining: List[str] = columns.copy()

        # Beam size for candidate evaluation per iteration
        top_k_default = 16
        beam_k = max(8, col_stop * 8) if col_stop and col_stop > 0 else top_k_default
        beam_k = int(min(top_k_default, beam_k))

        # Precompute vector of gid casted to uint64 when needed
        while remaining:
            # Candidate subset: top-k by base score not yet selected
            rem_sorted = sorted(remaining, key=lambda c: col_base_score.get(c, 0.0), reverse=True)
            k = min(len(rem_sorted), beam_k)

            best_col = None
            best_score = -1.0

            # Precompute left-shifted gid for combining with codes (as 64-bit)
            gid64_shifted = (gid.astype(np.uint64) << np.uint64(32))

            # Evaluate candidate incremental gains
            for col in rem_sorted[:k]:
                codes = col_codes[col].astype(np.uint64)
                pairs = gid64_shifted | codes
                # Unique count of pairs gives number of unique extended prefixes (S + col)
                unique_count = pd.unique(pairs).size
                dup_frac_ext = 1.0 - (unique_count / float(n_rows))
                # Gain approximated as average length times duplicates of extended prefix
                # Add run bonus as a small multiplier
                run_bonus = col_run_bonus[col]
                gain = dup_frac_ext * col_avg_len[col] * (0.5 + 0.5 * run_bonus)
                if gain > best_score:
                    best_score = gain
                    best_col = col

            if best_col is None:
                # Fallback when all gains are zero; choose by base score then avg length
                best_col = max(remaining, key=lambda c: (col_base_score.get(c, 0.0), col_avg_len.get(c, 0.0)))

            # Commit best column: update gid as factorization of pairs
            codes_best = col_codes[best_col].astype(np.uint64)
            pairs_best = (gid.astype(np.uint64) << np.uint64(32)) | codes_best
            gid = pd.factorize(pairs_best, sort=False)[0].astype(np.uint32)

            selected.append(best_col)
            remaining.remove(best_col)

        # Step 5: Reorder DataFrame columns according to selected order
        reordered = data.loc[:, selected]
        return reordered