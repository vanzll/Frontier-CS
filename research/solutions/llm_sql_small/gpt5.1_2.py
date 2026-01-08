import pandas as pd
import numpy as np
from itertools import permutations


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
        # Work on a string-typed copy to ensure consistent concatenation behavior
        df_work = df.astype(str)
        orig_cols = list(df_work.columns)

        # ----- Apply column merges (if any) -----
        if col_merge:
            # Normalize and clean merge groups
            groups_clean = []
            col_assigned = set()

            for group in col_merge:
                if not group:
                    continue
                names_resolved = []

                for g in group:
                    col_name = None
                    if isinstance(g, str):
                        if g in orig_cols:
                            col_name = g
                    elif isinstance(g, int):
                        # Try 0-based index
                        if 0 <= g < len(orig_cols):
                            col_name = orig_cols[g]
                        # Fallback to 1-based index if within range and 0-based failed
                        elif 1 <= g <= len(orig_cols):
                            col_name = orig_cols[g - 1]
                    if col_name is not None:
                        names_resolved.append(col_name)

                # Deduplicate while preserving order
                seen = set()
                unique = []
                for c in names_resolved:
                    if c not in seen:
                        seen.add(c)
                        unique.append(c)

                # Exclude columns that are already assigned to a previous group
                unassigned = [c for c in unique if c not in col_assigned]

                # Only merge if at least 2 distinct, unassigned columns remain
                if len(unassigned) >= 2:
                    groups_clean.append(unassigned)
                    col_assigned.update(unassigned)

            if groups_clean:
                group_cols_list = groups_clean
                group_first_col = []
                group_newnames = []
                group_series = []
                col_to_group = {}

                # Precompute group metadata and merged series
                for gid, cols in enumerate(group_cols_list):
                    # Determine the earliest column in original order for placement
                    indices = [orig_cols.index(c) for c in cols]
                    min_idx = min(indices)
                    first_col = orig_cols[min_idx]
                    group_first_col.append(first_col)

                    # Map each column to its group
                    for c in cols:
                        col_to_group[c] = gid

                    # Create merged column by concatenating strings in the specified group order
                    merged_series = df_work[cols].agg(''.join, axis=1)

                    # Generate a unique new column name
                    base_name = "__MERGED__" + "_" + "_".join(cols)
                    new_name = base_name
                    suffix = 1
                    existing_names = set(df_work.columns)
                    while new_name in existing_names or new_name in group_newnames:
                        suffix += 1
                        new_name = f"{base_name}_{suffix}"

                    group_newnames.append(new_name)
                    group_series.append(merged_series)

                # Build new DataFrame with merged columns replacing originals
                new_cols_data = {}
                new_cols_order = []

                for col in orig_cols:
                    gid = col_to_group.get(col)
                    if gid is None:
                        # Column not part of any merge group, keep as is
                        new_cols_data[col] = df_work[col]
                        new_cols_order.append(col)
                    else:
                        # Column is part of a merge group
                        # Insert merged column only at the first occurrence in original order
                        if col == group_first_col[gid]:
                            merged_name = group_newnames[gid]
                            new_cols_data[merged_name] = group_series[gid]
                            new_cols_order.append(merged_name)
                        # Skip other columns in the group

                df_work = pd.DataFrame(new_cols_data, index=df_work.index)
                df_work = df_work[new_cols_order]

        # ----- If 0 or 1 columns after merge, nothing to reorder -----
        cols = list(df_work.columns)
        M = len(cols)
        if M <= 1:
            return df_work

        N = len(df_work)

        # ----- Precompute column statistics: average length and equality probability -----
        avg_lens = []
        p_equals = []

        for col in cols:
            s = df_work[col]
            # Lengths of string values
            lengths = s.str.len().to_numpy()
            avg_len = float(lengths.mean()) if N > 0 else 0.0

            # Probability that two random rows have equal values in this column
            vc = s.value_counts(dropna=False)
            counts = vc.to_numpy(dtype=np.int64)
            if N > 1:
                equal_pairs = (counts * (counts - 1)).sum()
                p_equal = float(equal_pairs) / float(N * (N - 1))
            else:
                p_equal = 1.0

            avg_lens.append(avg_len)
            p_equals.append(p_equal)

        avg_lens = np.array(avg_lens, dtype=float)
        p_equals = np.array(p_equals, dtype=float)

        # ----- Determine best column order using approximate expected LCP model -----
        indices = list(range(M))

        if M <= 9:
            # Exhaustive search over all permutations
            best_perm = indices[:]
            best_score = -1.0

            for perm in permutations(indices):
                prod_equal = 1.0
                score = 0.0
                for idx in perm:
                    p_eq = p_equals[idx]
                    if prod_equal == 0.0:
                        break
                    score += prod_equal * p_eq * avg_lens[idx]
                    prod_equal *= p_eq
                if score > best_score:
                    best_score = score
                    best_perm = list(perm)
        else:
            # Fallback heuristic for larger M: sort by p_equal * avg_len descending
            scores = p_equals * avg_lens
            best_perm = sorted(indices, key=lambda i: scores[i], reverse=True)

        # ----- Reorder columns according to best_perm -----
        reordered_cols = [cols[i] for i in best_perm]
        df_reordered = df_work[reordered_cols]

        return df_reordered