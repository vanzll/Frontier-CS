import pandas as pd
import numpy as np
from joblib import Parallel, delayed
import multiprocessing

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
        if df.empty:
            return df

        df_processed = self._handle_merges(df.copy(), col_merge)

        df_str = df_processed.astype(str)
        N, M = df_str.shape

        if M <= 1:
            return df_processed

        sample_size = min(N, early_stop)
        df_sample = df_str.head(sample_size)

        stats = {c: df_sample[c].nunique() for c in df_sample.columns}
        
        high_card_cols = {
            c for c, n in stats.items()
            if (n / sample_size > distinct_value_threshold and n > 1)
        }
        low_card_cols = list(set(df_sample.columns) - high_card_cols)

        ordered_high_card = sorted(
            list(high_card_cols),
            key=lambda c: stats[c],
            reverse=True
        )

        ordered_low_card = self._greedy_search(
            df_sample,
            low_card_cols,
            stats,
            parallel
        )

        final_order = ordered_low_card + ordered_high_card
        return df_processed[final_order]

    def _handle_merges(self, df: pd.DataFrame, col_merge: list) -> pd.DataFrame:
        if not col_merge:
            return df

        cols_to_drop = set()
        for i, group in enumerate(col_merge):
            if not isinstance(group, list) or not group:
                continue

            existing_cols = [c for c in group if c in df.columns]
            if not existing_cols:
                continue

            new_col_name = f"_merged_col_{i}"
            df[new_col_name] = df[existing_cols].astype(str).agg("".join, axis=1)
            cols_to_drop.update(existing_cols)
        
        if cols_to_drop:
            df = df.drop(columns=list(cols_to_drop))
        return df

    def _greedy_search(
        self,
        df_sample: pd.DataFrame,
        low_card_cols: list,
        stats: dict,
        parallel: bool
    ) -> list:
        if not low_card_cols:
            return []

        low_card_cols.sort(key=lambda c: stats[c])
        
        remaining_cols = list(low_card_cols)
        ordered_cols = []

        first_col = remaining_cols.pop(0)
        ordered_cols.append(first_col)

        if not remaining_cols:
            return ordered_cols
        
        group_keys, _ = pd.factorize(df_sample[first_col].values, sort=False)
        
        try:
            num_cores = multiprocessing.cpu_count()
        except NotImplementedError:
            num_cores = 1

        for _ in range(len(remaining_cols)):
            if parallel and len(remaining_cols) > 1 and num_cores > 1:
                scores = Parallel(n_jobs=num_cores, prefer="threads")(
                    delayed(self._calculate_score)(df_sample, group_keys, c)
                    for c in remaining_cols
                )
                scored_cols = list(zip(scores, remaining_cols))
            else:
                scored_cols = [
                    (self._calculate_score(df_sample, group_keys, c), c)
                    for c in remaining_cols
                ]

            scored_cols.sort(key=lambda x: (x[0], stats[x[1]]))
            
            best_col = scored_cols[0][1]
            
            ordered_cols.append(best_col)
            remaining_cols.remove(best_col)

            if remaining_cols:
                combined_keys = list(zip(group_keys, df_sample[best_col].values))
                group_keys, _ = pd.factorize(combined_keys, sort=False)
        
        return ordered_cols

    @staticmethod
    def _calculate_score(df_sample: pd.DataFrame, group_keys: np.ndarray, col: str) -> int:
        combined_keys = list(zip(group_keys, df_sample[col].values))
        _, uniques = pd.factorize(combined_keys, sort=False)
        return len(uniques)