import pandas as pd
import numpy as np
from typing import List, Optional


class Solution:
    def _apply_col_merges(self, df: pd.DataFrame, col_merge: Optional[List[List[str]]]) -> pd.DataFrame:
        if not col_merge:
            return df

        working_df = df
        for group in col_merge:
            if not group:
                continue
            existing_cols = [c for c in group if c in working_df.columns]
            if len(existing_cols) <= 1:
                continue

            base_name = existing_cols[0]
            new_name = f"{base_name}__merged"
            while new_name in working_df.columns:
                new_name += "_m"

            # Concatenate specified columns as strings without separators
            working_df[new_name] = working_df[existing_cols].astype(str).agg(''.join, axis=1)
            working_df = working_df.drop(columns=existing_cols)

        return working_df

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
        # Work on a copy to avoid mutating the input DataFrame
        working_df = df.copy()

        # Apply column merges (if any)
        working_df = self._apply_col_merges(working_df, col_merge)

        n_rows, n_cols = working_df.shape
        if n_cols <= 1 or n_rows == 0:
            return working_df

        # Determine sample size for statistics to keep runtime low
        max_sample_size = 12000
        sample_n = min(n_rows, early_stop, max_sample_size)

        sample_df = working_df.iloc[:sample_n].astype(str)
        col_names = list(working_df.columns)
        orig_pos = {name: idx for idx, name in enumerate(col_names)}

        col_metrics = {}

        for col in col_names:
            s = sample_df[col]
            if s.empty:
                col_metrics[col] = (0, 0)
                continue

            len_series = s.str.len()
            total_len = int(len_series.sum())

            vc = s.value_counts(dropna=False)
            freqs = vc.values.astype(np.int64)
            vals = vc.index.tolist()

            if len(vals) == 0:
                col_metrics[col] = (0, total_len)
                continue

            lens_vals = np.fromiter((len(v) for v in vals), dtype=np.int64, count=len(vals))
            metric = int((freqs * freqs * lens_vals).sum())

            col_metrics[col] = (metric, total_len)

        # Sort columns: higher expected pairwise LCP contribution first
        sorted_cols = sorted(
            col_names,
            key=lambda c: (-col_metrics[c][0], -col_metrics[c][1], orig_pos[c])
        )

        return working_df[sorted_cols]