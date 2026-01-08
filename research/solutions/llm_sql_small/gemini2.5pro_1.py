import pandas as pd
import numpy as np
from joblib import Parallel, delayed

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
        
        processed_df, columns_to_order = self._handle_merges(df, col_merge)

        if len(columns_to_order) <= 1:
            return processed_df

        df_sample = processed_df.head(early_stop)
        df_sample_str = df_sample[columns_to_order].astype(str)

        high_card_cols, low_card_cols = self._classify_columns(
            df_sample_str, columns_to_order, distinct_value_threshold
        )

        ordered_low_card_cols = self._greedy_search(
            df_sample_str, low_card_cols, parallel
        )
        
        ordered_high_card_cols = self._sort_by_cardinality(
            df_sample_str, high_card_cols
        )

        final_order = ordered_low_card_cols + ordered_high_card_cols
        
        return processed_df[final_order]

    @staticmethod
    def _handle_merges(df: pd.DataFrame, col_merge: list) -> (pd.DataFrame, list):
        if not col_merge:
            return df, list(df.columns)

        processed_df = df.copy()
        cols_to_drop = set()

        for i, group in enumerate(col_merge):
            valid_group = [col for col in group if col in processed_df.columns]
            if not valid_group:
                continue
            
            merged_col_name = f"__merged_{i}_" + "_".join(map(str, valid_group))
            
            processed_df[merged_col_name] = processed_df[valid_group].astype(str).agg(''.join, axis=1)
            cols_to_drop.update(valid_group)

        if cols_to_drop:
            processed_df.drop(columns=list(cols_to_drop), inplace=True)
        
        return processed_df, list(processed_df.columns)

    @staticmethod
    def _classify_columns(
        df_sample: pd.DataFrame, columns: list, threshold: float
    ) -> (list, list):
        high_card_cols = []
        low_card_cols = []
        num_rows_sample = len(df_sample)
        if num_rows_sample == 0:
            return [], columns

        for col in columns:
            distinct_count = df_sample[col].nunique()
            if distinct_count == 1:
                distinct_ratio = 0.0
            else:
                distinct_ratio = distinct_count / num_rows_sample
            
            if distinct_ratio > threshold:
                high_card_cols.append(col)
            else:
                low_card_cols.append(col)
        
        return high_card_cols, low_card_cols

    @staticmethod
    def _get_score(df_sample: pd.DataFrame, ordered_cols: list, candidate_col: str) -> int:
        if not ordered_cols:
            return df_sample[candidate_col].nunique()
        else:
            return len(df_sample[ordered_cols + [candidate_col]].drop_duplicates())

    @classmethod
    def _greedy_search(
        cls, df_sample: pd.DataFrame, cols: list, parallel: bool
    ) -> list:
        if not cols:
            return []
            
        remaining_cols = cols.copy()
        ordered_cols = []
        
        for _ in range(len(cols)):
            if not remaining_cols:
                break
            
            if parallel and len(remaining_cols) > 1:
                scores = Parallel(n_jobs=-1, prefer="threads")(
                    delayed(cls._get_score)(df_sample, ordered_cols, c) 
                    for c in remaining_cols
                )
            else:
                scores = [cls._get_score(df_sample, ordered_cols, c) for c in remaining_cols]
            
            if not scores:
                break

            best_col_idx = np.argmin(scores)
            best_col = remaining_cols.pop(best_col_idx)
            ordered_cols.append(best_col)
            
        return ordered_cols
        
    @staticmethod
    def _sort_by_cardinality(df_sample: pd.DataFrame, cols: list) -> list:
        if not cols or len(cols) <= 1:
            return cols

        cardinalities = df_sample[cols].nunique()
        return cardinalities.sort_values(ascending=False).index.tolist()