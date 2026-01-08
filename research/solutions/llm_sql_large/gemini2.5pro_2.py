import pandas as pd
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

def _evaluate_candidate(prefix, col, df_sample):
    """
    Helper function for multiprocessing. It must be defined at the top level.
    Calculates the number of unique groups for a new potential prefix.
    """
    new_prefix = prefix + [col]
    try:
        # Using itertuples and set is generally faster than drop_duplicates
        score = len(set(df_sample[new_prefix].itertuples(index=False, name=None)))
    except Exception:
        # Fallback for complex types that might not be hashable
        score = df_sample[new_prefix].drop_duplicates().shape[0]
    return (new_prefix, score)

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
        original_df = df
        
        # 1. Handle column merges
        if col_merge:
            df = df.copy()
            new_cols_data = {}
            cols_to_drop = set()
            for i, group in enumerate(col_merge):
                if not group:
                    continue
                new_col_name = f"__merged_{i}__"
                new_cols_data[new_col_name] = df[group].astype(str).apply("".join, axis=1)
                cols_to_drop.update(group)
            
            df.drop(columns=list(cols_to_drop), inplace=True)
            df = pd.concat([df, pd.DataFrame(new_cols_data, index=df.index)], axis=1)

        # 2. Convert all columns to string type for consistent processing
        df_str = df.astype(str)

        # 3. Sample the DataFrame for determining the column order
        sample_size = min(len(df_str), early_stop)
        df_sample = df_str.head(sample_size)

        if df_sample.empty:
            return original_df[df.columns]

        # 4. Identify high-cardinality columns to be moved to the end
        sample_len = len(df_sample)
        all_cols = list(df_sample.columns)
        col_cardinalities = {c: df_sample[c].nunique() for c in all_cols}
        
        high_card_cols = {
            c for c, card in col_cardinalities.items()
            if card > 1 and (card / sample_len) > distinct_value_threshold
        }
        high_card_cols.update({c for c, card in col_cardinalities.items() if card == sample_len and sample_len > 1})

        cols_to_order = [c for c in all_cols if c not in high_card_cols]

        if not cols_to_order:
            cols_to_order = all_cols
            high_card_cols = set()

        # 5. Beam Search to find the best initial sequence of columns
        beam = [([], 1)]  # List of (prefix_list, score)
        beam_width = row_stop
        search_depth = min(col_stop, len(cols_to_order))

        for _ in range(search_depth):
            if not beam: break
            
            all_potential_candidates = []
            for prefix, _ in beam:
                remaining_cols = [c for c in cols_to_order if c not in prefix]
                for c in remaining_cols:
                    all_potential_candidates.append((prefix, c))
            
            if not all_potential_candidates: break
            
            candidates = []
            use_parallel = parallel and len(all_potential_candidates) > 1 and os.cpu_count() is not None and os.cpu_count() > 1
            
            if use_parallel:
                with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
                    futures = {executor.submit(_evaluate_candidate, prefix, c, df_sample): (prefix, c) for prefix, c in all_potential_candidates}
                    for future in as_completed(futures):
                        try:
                            candidates.append(future.result())
                        except Exception:
                            pass
            else:
                for prefix, c in all_potential_candidates:
                    candidates.append(_evaluate_candidate(prefix, c, df_sample))
            
            if not candidates: break

            candidates.sort(key=lambda x: x[1])
            beam = candidates[:beam_width]

        # 6. Assemble the final column order
        best_prefix = beam[0][0] if (beam and beam[0][0]) else []

        remaining_low_card_cols = [c for c in cols_to_order if c not in best_prefix]
        remaining_low_card_cols.sort(key=lambda c: col_cardinalities.get(c, float('inf')))
        
        sorted_high_card_cols = sorted(list(high_card_cols), key=lambda c: col_cardinalities.get(c, float('inf')))

        final_order = best_prefix + remaining_low_card_cols + sorted_high_card_cols
        
        if len(final_order) != len(all_cols):
             low_card_cols_sorted = sorted([c for c in all_cols if c not in high_card_cols], key=lambda c: col_cardinalities.get(c, float('inf')))
             final_order = low_card_cols_sorted + sorted_high_card_cols

        # 7. Return the original DataFrame with reordered columns
        return original_df[final_order]