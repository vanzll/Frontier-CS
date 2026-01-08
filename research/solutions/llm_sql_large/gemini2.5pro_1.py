import pandas as pd
from joblib import Parallel, delayed

def _calculate_new_groups(df_sample, ordered_cols, next_col):
    """
    Calculates the number of unique groups in a DataFrame based on a list of columns.
    This function is defined at the top level to be easily picklable by joblib.
    """
    cols_to_check = ordered_cols + [next_col]
    return df_sample[cols_to_check].drop_duplicates().shape[0]

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
        """
        if col_merge:
            df_merged = df.copy()
            original_cols_to_drop = set()
            for i, group in enumerate(col_merge):
                valid_group = [c for c in group if c in df_merged.columns]
                if not valid_group:
                    continue
                
                merged_col_name = f"_merged_{i}"
                df_merged[merged_col_name] = df_merged[valid_group].astype(str).agg(''.join, axis=1)
                original_cols_to_drop.update(valid_group)
            
            df_merged = df_merged.drop(columns=[c for c in original_cols_to_drop if c in df_merged.columns])
        else:
            df_merged = df

        df_str = df_merged.astype(str)
        all_columns = list(df_str.columns)

        if len(all_columns) <= 1:
            return df_merged

        df_sample = df_str.head(min(early_stop, len(df_str)))
        n_sample = len(df_sample)
        if n_sample == 0:
            return df_merged

        nunique_map = {c: df_sample[c].nunique() for c in all_columns}

        low_card_cols = [
            c for c in all_columns 
            if nunique_map.get(c, n_sample + 1) / n_sample <= distinct_value_threshold
        ]
        high_card_cols = [c for c in all_columns if c not in low_card_cols]

        ordered_low_card = self._order_columns_greedy(
            df_sample, low_card_cols, nunique_map, col_stop, row_stop, parallel
        )
        
        high_card_cols.sort(key=lambda c: nunique_map.get(c, float('inf')))

        final_order = ordered_low_card + high_card_cols
        
        if set(final_order) != set(df_merged.columns):
             all_cols_sorted = sorted(list(df_merged.columns), key=lambda c: nunique_map.get(c, float('inf')))
             return df_merged[all_cols_sorted]

        return df_merged[final_order]

    def _order_columns_greedy(
        self, df_sample: pd.DataFrame, available_cols: list, nunique_map: dict,
        col_stop: int, row_stop: int, parallel: bool
    ) -> list:
        if not available_cols:
            return []

        n_sample = len(df_sample)
        remaining_cols = sorted(available_cols, key=lambda c: nunique_map.get(c, float('inf')))
        ordered_cols = []
        
        greedy_phase_len = min(col_stop, len(remaining_cols))
        
        while len(ordered_cols) < greedy_phase_len:
            if not remaining_cols:
                break
            
            if ordered_cols and row_stop > 0:
                num_groups = df_sample[ordered_cols].drop_duplicates().shape[0]
                if num_groups * row_stop > n_sample:
                    break
            
            if len(remaining_cols) == 1:
                ordered_cols.append(remaining_cols.pop(0))
                break

            if parallel:
                group_counts = Parallel(n_jobs=-1, prefer="threads")(
                    delayed(_calculate_new_groups)(df_sample, ordered_cols, next_col)
                    for next_col in remaining_cols
                )
                scores = sorted(zip(group_counts, [nunique_map.get(c, float('inf')) for c in remaining_cols], remaining_cols))
            else:
                scores = []
                for col in remaining_cols:
                    num_groups = _calculate_new_groups(df_sample, ordered_cols, col)
                    scores.append((num_groups, nunique_map.get(col, float('inf')), col))
                scores.sort()
            
            if not scores:
                break
            
            best_next_col = scores[0][2]
            ordered_cols.append(best_next_col)
            remaining_cols.remove(best_next_col)
        
        if remaining_cols:
            ordered_cols.extend(remaining_cols)
            
        return ordered_cols