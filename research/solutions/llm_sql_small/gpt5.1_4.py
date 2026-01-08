import pandas as pd
import numpy as np


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
        """
        Reorder columns in the DataFrame to maximize prefix hit rate.
        
        Args:
            df: Input DataFrame to optimize
            early_stop: Early stopping parameter (unused in this heuristic)
            row_stop: Row stopping parameter (unused in this heuristic)
            col_stop: Column stopping parameter (unused in this heuristic)
            col_merge: List of column groups to merge (columns in each group are merged into one)
            one_way_dep: List of one-way dependencies (not used in this variant)
            distinct_value_threshold: Threshold for distinct values (used to penalize very high-cardinality columns)
            parallel: Whether to use parallel processing (unused in this heuristic)
        
        Returns:
            DataFrame with reordered (and merged) columns
        """
        if df is None or df.empty:
            return df

        # Work on a copy to avoid mutating the original DataFrame
        work_df = df.copy()

        # Step 1: Apply column merges if specified
        work_df = self._apply_col_merges(work_df, col_merge)

        # If there is 0 or 1 column after merging, nothing to reorder
        if work_df.shape[1] <= 1:
            return work_df

        # Step 2: Compute heuristic scores for columns
        scores = self._compute_column_scores(
            work_df,
            distinct_value_threshold=distinct_value_threshold,
        )

        # Step 3: Sort columns by descending score, tie-broken by original position
        original_positions = {col: idx for idx, col in enumerate(work_df.columns)}
        ordered_cols = sorted(
            work_df.columns,
            key=lambda c: (-scores.get(c, 0.0), original_positions[c]),
        )

        return work_df[ordered_cols]

    def _apply_col_merges(self, df: pd.DataFrame, col_merge: list) -> pd.DataFrame:
        """
        Apply column merges as specified by col_merge.
        Each group in col_merge is a list of column names to concatenate into one column.
        The merged column takes the name of the first existing column in the group.
        """
        if not col_merge:
            return df

        merged_df = df.copy()

        for group in col_merge:
            # Ensure group is iterable
            if not isinstance(group, (list, tuple)):
                group = [group]

            # Keep only columns that currently exist in the DataFrame
            cols_in_group = [c for c in group if c in merged_df.columns]
            if len(cols_in_group) <= 1:
                # Nothing to merge (either no columns or a single column)
                continue

            merged_name = cols_in_group[0]
            # Compute merged string column
            merged_series = merged_df[cols_in_group].astype(str).agg("".join, axis=1)

            # Insert merged column at the position of merged_name (first col in group)
            insert_pos = merged_df.columns.get_loc(merged_name)

            # Drop the original columns in the group
            merged_df = merged_df.drop(columns=cols_in_group)

            # Insert the merged column
            cols = list(merged_df.columns)
            cols.insert(insert_pos, merged_name)
            merged_df[merged_name] = merged_series
            merged_df = merged_df[cols]

        return merged_df

    def _compute_column_scores(
        self,
        df: pd.DataFrame,
        distinct_value_threshold: float,
    ) -> dict:
        """
        Compute a heuristic score for each column that estimates its usefulness
        as an early prefix for maximizing KV-cache hit rate.

        Heuristic:
            score(col) ≈ (sum(freq(value)^2) / N^2) * avg_string_length(col) * penalty

        where:
            - freq(value) are value frequencies of the column
            - N is number of rows
            - penalty down-weights very high-cardinality columns using distinct_value_threshold
        """
        n_rows = len(df)
        if n_rows == 0:
            # Degenerate case: no rows; all scores are zero
            return {col: 0.0 for col in df.columns}

        scores = {}
        n_rows_float = float(n_rows)
        denom = n_rows_float * n_rows_float

        # Clamp threshold to (0,1); if invalid, disable penalty
        use_threshold_penalty = 0.0 < distinct_value_threshold < 1.0

        for col in df.columns:
            series = df[col]

            # Frequency distribution (including NaN as a category)
            vc = series.value_counts(dropna=False)
            if vc.empty:
                scores[col] = 0.0
                continue

            counts = vc.values.astype(np.int64)
            freq2_sum = float((counts * counts).sum())
            norm_share = freq2_sum / denom  # in [1/N, 1]

            # Distinct ratio is number of unique values over N
            distinct_ratio = float(len(vc)) / n_rows_float

            # Average string length
            col_str = series.astype(str)
            avg_len = float(col_str.str.len().mean())

            # Distinctness penalty: linearly penalize distinct_ratio above threshold
            if use_threshold_penalty:
                if distinct_ratio <= distinct_value_threshold:
                    distinct_penalty = 1.0
                else:
                    # Map distinct_ratio in [threshold,1] -> penalty in [1,0]
                    if distinct_ratio >= 1.0:
                        distinct_penalty = 0.0
                    else:
                        distinct_penalty = max(
                            0.0,
                            1.0
                            - (distinct_ratio - distinct_value_threshold)
                            / (1.0 - distinct_value_threshold),
                        )
            else:
                distinct_penalty = 1.0

            score = norm_share * avg_len * distinct_penalty
            scores[col] = score

        return scores