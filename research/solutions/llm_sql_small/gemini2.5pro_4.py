import pandas as pd
from joblib import Parallel, delayed

def _calculate_score(col_name: str, 
                     col_series: pd.Series, 
                     current_prefixes: tuple, 
                     nunique_for_col: int):
    """
    Helper function for parallel execution. Calculates the number of distinct prefixes
    when a new column's values are appended.
    """
    col_values = col_series.to_list()
    new_prefixes = tuple(p + (v,) for p, v in zip(current_prefixes, col_values))
    distinct_count = len(set(new_prefixes))
    return (distinct_count, nunique_for_col, col_name, new_prefixes)

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
        
        work_df = df.copy()

        if col_merge:
            for group in col_merge:
                if len(group) > 1 and all(c in work_df.columns for c in group):
                    new_col_name = '_'.join(str(c) for c in group)
                    work_df[new_col_name] = work_df[group].astype(str).agg(''.join, axis=1)
                    work_df.drop(columns=group, inplace=True)

        if work_df.shape[1] <= 1:
            return work_df

        sample_size = min(len(work_df), early_stop)
        df_sample = work_df.head(sample_size).astype(str)
        
        all_cols = list(df_sample.columns)

        nuniques = {col: df_sample[col].nunique() for col in all_cols}

        high_card_cols = []
        low_card_cols = []
        for col in all_cols:
            if nuniques[col] / sample_size > distinct_value_threshold:
                high_card_cols.append(col)
            else:
                low_card_cols.append(col)

        ordered_prefix = []
        search_cols = list(low_card_cols)
        greedy_steps = min(col_stop, len(search_cols))
        
        current_prefixes = tuple(() for _ in range(sample_size))

        n_jobs = -1 if parallel and len(search_cols) > 1 else 1
        
        for _ in range(greedy_steps):
            if not search_cols:
                break
            
            results = Parallel(n_jobs=n_jobs, prefer="threads")(
                delayed(_calculate_score)(col, df_sample[col], current_prefixes, nuniques[col])
                for col in search_cols
            )
            
            results.sort(key=lambda res: (res[0], res[1]))
            
            _distinct_count, _nunique, best_col, best_prefixes = results[0]
            
            current_prefixes = best_prefixes
            ordered_prefix.append(best_col)
            search_cols.remove(best_col)

        remaining_low_card_cols = search_cols
        remaining_low_card_cols.sort(key=lambda c: nuniques[c])
        high_card_cols.sort(key=lambda c: nuniques[c])

        final_order = ordered_prefix + remaining_low_card_cols + high_card_cols
        
        return work_df[final_order]