import pandas as pd
import numpy as np
import multiprocessing as mp

# This helper function must be defined at the top level of the module
# for it to be pickleable by the multiprocessing library.
def _calculate_score_static(col_values, group_ids):
    """
    Calculates the grouping score for a candidate column.
    The score is the sum of squares of the sizes of the new groups that would be
    formed by adding this column to the current set of grouping columns.
    A higher score indicates a better grouping (more rows clustered together).
    
    Args:
        col_values (np.ndarray): The values of the candidate column.
        group_ids (np.ndarray): The current group IDs for each row.
        
    Returns:
        int: The calculated score.
    """
    new_group_counts = {}
    for i in range(len(group_ids)):
        # Create a new key based on the old group and the new column's value
        key = (group_ids[i], col_values[i])
        new_group_counts[key] = new_group_counts.get(key, 0) + 1
    
    # The score is the sum of squares of the counts of each new group.
    # This is a proxy for maximizing the number of pairs of rows that match
    # on the extended prefix, which correlates with maximizing LCP.
    score = sum(v * v for v in new_group_counts.values())
    return score

class Solution:
    """
    Implements a solution to reorder DataFrame columns to maximize a simulated
    KV-cache hit rate for LLM inference.
    """
    
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
        Reorders columns in the DataFrame to maximize prefix hit rate.

        The strategy is a greedy algorithm that iteratively builds a column
        permutation. At each step, it selects the column that best improves
        the clustering of rows, measured by a score that is proportional to
        the sum of squares of group sizes. This heuristic aims to create a
        lexicographical sort order that maximizes overlap between adjacent rows.

        The process includes:
        1.  Merging specified columns into new single columns.
        2.  Separating high-cardinality columns (like IDs), which are poor
            candidates for prefixes, to be placed at the end of the order.
        3.  Running a parallelized greedy search for a specified number of steps
            (`col_stop`) to determine the optimal prefix of the column order.
        4.  Ordering the remaining columns based on a simpler heuristic (cardinality)
            which continues the greedy principle.
        5.  Using a sample of the data (`early_stop`) for efficiency if the
            dataset is large.
        """
        
        df_processed = df.copy()

        # 1. Handle column merges if specified
        if col_merge:
            for merge_group in col_merge:
                if len(merge_group) > 1 and all(c in df_processed.columns for c in merge_group):
                    new_col_name = "_".join(merge_group)
                    df_processed[new_col_name] = df_processed[merge_group].astype(str).agg("".join, axis=1)
                    df_processed = df_processed.drop(columns=merge_group)

        all_cols = df_processed.columns.tolist()
        
        if len(all_cols) <= 1:
            return df_processed

        # 2. Use a sample for statistics if specified and necessary
        n_rows_total = len(df_processed)
        sample_size = min(n_rows_total, early_stop)
        if sample_size < n_rows_total:
            df_sample = df_processed.sample(n=sample_size, random_state=42)
        else:
            df_sample = df_processed
        
        n_rows_sample = len(df_sample)
        if n_rows_sample == 0:
            return df_processed
            
        df_str = df_sample.astype(str)

        # 3. Separate high-cardinality columns
        high_card_cols = [
            c for c in all_cols 
            if df_sample[c].nunique() / n_rows_sample > distinct_value_threshold
        ]
        low_card_cols = [c for c in all_cols if c not in high_card_cols]
        
        if not low_card_cols:
            final_order = sorted(high_card_cols)
            return df_processed[final_order]

        # 4. Greedy search for the best prefix of columns
        permutation = []
        unselected_cols = low_card_cols.copy()
        
        group_ids = np.zeros(n_rows_sample, dtype=np.int32)
        
        num_greedy_steps = min(len(low_card_cols), col_stop)

        pool = None
        if parallel and mp.cpu_count() > 1:
            pool = mp.Pool(processes=min(mp.cpu_count(), 8))

        for _ in range(num_greedy_steps):
            if not unselected_cols:
                break
            
            tasks = [(df_str[c].values, group_ids) for c in unselected_cols]
            
            if pool:
                scores = pool.starmap(_calculate_score_static, tasks)
                results = list(zip(scores, unselected_cols))
            else:
                results = [(_calculate_score_static(task[0], task[1]), col) for task, col in zip(tasks, unselected_cols)]

            if not results:
                break
                
            best_score, best_col = max(results)
            
            permutation.append(best_col)
            unselected_cols.remove(best_col)

            # Update group_ids for the next iteration using a fast vectorised approach
            best_col_values = df_str[best_col].values
            # Combining previous group_ids with new column values to form new groups.
            # String conversion and concatenation is a robust way to create unique keys
            # for pd.factorize, which is highly optimized for this task.
            composite_keys = pd.Series(group_ids).astype(str).values + '_' + best_col_values
            _, group_ids = pd.factorize(composite_keys, sort=False)
            group_ids = group_ids.astype(np.int32)

        if pool:
            pool.close()
            pool.join()
            
        # 5. Order remaining columns by a simpler heuristic (cardinality)
        sorted_remaining_low_card = []
        if unselected_cols:
            cardinalities = df_sample[unselected_cols].nunique().sort_values()
            sorted_remaining_low_card = cardinalities.index.tolist()

        # 6. Combine all parts for the final order
        sorted_high_card = sorted(high_card_cols)
        final_order = permutation + sorted_remaining_low_card + sorted_high_card
        
        # Sanity check to ensure all columns are accounted for
        if set(final_order) != set(all_cols):
             all_cardinalities = df_processed.nunique().sort_values()
             final_order = all_cardinalities.index.tolist()

        return df_processed[final_order]