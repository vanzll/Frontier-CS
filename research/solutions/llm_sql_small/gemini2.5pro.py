import pandas as pd
import numpy as np
from joblib import Parallel, delayed
import os

class Solution:
    """
    Solution class to reorder columns of a DataFrame to maximize KV-cache hit rate.
    """

    def _get_incremental_score(self, col: str, current_perm: list, df_str: pd.DataFrame) -> float:
        """
        Calculates the incremental score for adding a column to the current permutation.
        The score is a proxy for the LCP gain, computed by summing up the
        potential LCP contribution within groups formed by the current permutation.
        """
        score = 0
        if not current_perm:
            # For the first column, groups are not needed.
            counts = df_str[col].value_counts()
            # The logic (count - 1) * len(val) approximates the LCP sum
            # if we were to sort by this single column.
            for val, count in counts.items():
                if count > 1:
                    score += (count - 1) * len(str(val))
        else:
            # For subsequent columns, calculate score within each group
            # defined by the columns in `current_perm`.
            try:
                groups = df_str.groupby(current_perm)
                for _, group_indices in groups.groups.items():
                    if len(group_indices) > 1:
                        # Get value counts for the candidate column `col` within the current group
                        counts = df_str.loc[group_indices, col].value_counts()
                        for val, count in counts.items():
                            if count > 1:
                                score += (count - 1) * len(str(val))
            except KeyError:
                return 0.0

        return float(score)

    def _find_best_perm_greedy(self, cols: list, df_str: pd.DataFrame, parallel: bool) -> list:
        """
        Finds the best permutation of columns using a sequential greedy algorithm.
        At each step, it selects the column that provides the maximum incremental score.
        """
        num_cols = len(cols)
        perm = []
        rem_cols = list(cols)

        for _ in range(num_cols):
            if not rem_cols:
                break

            if parallel:
                try:
                    # Use joblib for parallel execution. n_jobs=-1 uses all available CPU cores.
                    # 'threads' backend is chosen to avoid data serialization overhead with pandas DataFrames.
                    scores = Parallel(n_jobs=-1, prefer="threads")(
                        delayed(self._get_incremental_score)(c, perm, df_str) for c in rem_cols
                    )
                except Exception:
                    # Fallback to sequential execution if parallel processing fails
                    scores = [self._get_incremental_score(c, perm, df_str) for c in rem_cols]
            else:
                scores = [self._get_incremental_score(c, perm, df_str) for c in rem_cols]

            if not scores:
                break
            
            best_idx = np.argmax(scores)
            best_col = rem_cols.pop(best_idx)
            perm.append(best_col)

        return perm

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
        """
        Reorder columns in the DataFrame to maximize prefix hit rate.
        
        Args:
            df: Input DataFrame to optimize
            early_stop: Early stopping parameter (default: 100000)
            row_stop: Row stopping parameter (default: 4)
            col_stop: Column stopping parameter (default: 2)
            col_merge: List of column groups to merge (columns in each group are merged into one)
            one_way_dep: List of one-way dependencies (not used in this variant)
            distinct_value_threshold: Threshold for distinct values (default: 0.7)
            parallel: Whether to use parallel processing (default: True)
        
        Returns:
            DataFrame with reordered columns (same rows, different column order)
        """
        if df.empty:
            return df

        # 1. Handle column merges
        working_df = df
        if col_merge:
            working_df = df.copy()
            cols_to_drop = set()
            for group in col_merge:
                if not isinstance(group, list) or len(group) < 2:
                    continue
                
                group = [c for c in group if c in working_df.columns]
                if len(group) < 2:
                    continue

                new_col_name = '_'.join(map(str, group))
                
                str_df_group = working_df[group].astype(str)
                # Faster concatenation using numpy
                working_df[new_col_name] = pd.Series(str_df_group.values.sum(axis=1), index=working_df.index)
                
                for col in group:
                    cols_to_drop.add(col)
            
            working_df = working_df.drop(columns=list(cols_to_drop), errors='ignore')

        # 2. Use a subset of the data for analysis if it's too large.
        if len(working_df) > early_stop:
            analytic_df = working_df.sample(n=early_stop, random_state=42)
        else:
            analytic_df = working_df
            
        cols = list(analytic_df.columns)
        if len(cols) <= 1:
            return working_df

        # 3. Partition columns into low-cardinality (LC) and high-cardinality (HC)
        lc_cols = []
        hc_cols = []
        
        if len(analytic_df) > 0:
            unique_counts = {c: analytic_df[c].nunique() for c in cols}
            for col in cols:
                cardinality_ratio = unique_counts[col] / len(analytic_df)
                if cardinality_ratio >= distinct_value_threshold:
                    hc_cols.append(col)
                else:
                    lc_cols.append(col)
        else: # Handle empty analytic_df
            lc_cols = cols

        # 4. Find the best permutation for LC columns using the greedy approach
        df_str_lc = analytic_df[lc_cols].astype(str)
        perm_lc = self._find_best_perm_greedy(lc_cols, df_str_lc, parallel)

        # 5. Order HC columns based on their cardinality (ascending)
        if hc_cols:
            if len(analytic_df) > 0:
                hc_cardinalities = {c: unique_counts[c] for c in hc_cols}
                perm_hc = sorted(hc_cols, key=lambda c: hc_cardinalities.get(c, 0))
            else: # Fallback if analytic_df is empty
                perm_hc = sorted(hc_cols)
        else:
            perm_hc = []

        # 6. Combine permutations
        final_perm = perm_lc + perm_hc

        # 7. Return the processed DataFrame with reordered columns
        return working_df[final_perm]