import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from collections import Counter
import math

# This function must be defined at the top level to be picklable by joblib
def _calculate_cond_entropy_for_col(col_name, data, cols_map, group_ids, N, group_sizes):
    """
    Calculates the conditional entropy of a column given the current grouping.
    H(C | P) = sum_{g in P} P(g) * H(C | g)
    """
    col_idx = cols_map[col_name]
    
    # Count occurrences of (group_id, value)
    new_group_counts = Counter()
    col_values = data[:, col_idx]
    for i in range(N):
        key = (group_ids[i], col_values[i])
        new_group_counts[key] += 1
        
    cond_entropy = 0.0
    
    # Using the formula: H(C|P) = - sum_{g,v} p(g,v) log2(p(v|g))
    # which is sum_{g,v} - (count(g,v)/N) * log2(count(g,v)/size(g))
    for (gid, _), count in new_group_counts.items():
        group_size = group_sizes[gid]
        if group_size > 1: # No entropy gain if group size is 1 or 0
            # p_v_g can be 1.0, leading to log(1)=0, which is correct.
            p_v_g = count / group_size
            cond_entropy -= (count / N) * math.log2(p_v_g)
            
    return col_name, cond_entropy, new_group_counts

class Solution:
    def _find_optimal_order(self, df: pd.DataFrame, parallel: bool) -> list:
        if df.empty or df.shape[1] == 0:
            return []

        N, M = df.shape
        data = df.to_numpy()
        cols_map = {name: i for i, name in enumerate(df.columns)}
        
        remaining_cols = df.columns.tolist()
        ordered_cols = []
        
        group_ids = np.zeros(N, dtype=np.int32)
        num_groups = 1
        
        for _ in range(M):
            if not remaining_cols:
                break

            group_sizes = np.bincount(group_ids, minlength=num_groups)

            best_col_name = None
            min_cond_entropy = float('inf')
            
            if parallel and len(remaining_cols) > 1:
                results = Parallel(n_jobs=-1, prefer="threads")(
                    delayed(_calculate_cond_entropy_for_col)(
                        c, data, cols_map, group_ids, N, group_sizes
                    ) for c in remaining_cols
                )
                
                best_col_counts = None
                for col_name, cond_entropy, new_group_counts in results:
                    if cond_entropy < min_cond_entropy:
                        min_cond_entropy = cond_entropy
                        best_col_name = col_name
                        best_col_counts = new_group_counts
            else:
                best_col_counts = None
                for col_name in remaining_cols:
                    _, cond_entropy, new_group_counts = _calculate_cond_entropy_for_col(
                        col_name, data, cols_map, group_ids, N, group_sizes
                    )
                    if cond_entropy < min_cond_entropy:
                        min_cond_entropy = cond_entropy
                        best_col_name = col_name
                        best_col_counts = new_group_counts

            if best_col_name is None: # Fallback for single remaining column or other edge cases
                best_col_name = remaining_cols[0]
                _, _, best_col_counts = _calculate_cond_entropy_for_col(
                        best_col_name, data, cols_map, group_ids, N, group_sizes
                )

            ordered_cols.append(best_col_name)
            remaining_cols.remove(best_col_name)

            if not remaining_cols:
                break
            
            # Update group_ids for the next iteration using the best column
            col_idx = cols_map[best_col_name]
            # Vectorized update using pandas.factorize for speed
            keys = list(zip(group_ids, data[:, col_idx]))
            new_group_ids, _ = pd.factorize(keys, sort=False)
            group_ids = new_group_ids
            num_groups = (group_ids.max() + 1) if N > 0 and group_ids.size > 0 else 0

        return ordered_cols

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
        
        df_processed = df.copy()

        if col_merge:
            cols_to_drop = set()
            for i, group in enumerate(col_merge):
                if not group or len(group) == 0: continue
                
                # Ensure all columns in the group exist
                valid_group = [c for c in group if c in df_processed.columns]
                if not valid_group: continue
                
                cols_to_drop.update(valid_group)
                new_col_name = f"_merged_{i}_{'_'.join(valid_group)}"
                df_processed[new_col_name] = df_processed[valid_group].astype(str).agg(''.join, axis=1)
            
            df_processed = df_processed.drop(columns=list(cols_to_drop))

        if df_processed.empty:
            return df_processed

        # Use a sample for analysis to manage runtime, though defaults might use full dataset
        sample_size = min(early_stop, len(df_processed))
        df_sample = df_processed.head(sample_size)
        df_sample = df_sample.astype(str)
        N_sample = len(df_sample)
        
        all_cols = df_sample.columns.tolist()
        good_cols = []
        bad_cols = []
        
        if N_sample > 0:
            # Partition columns into "good" (low cardinality) and "bad" (high cardinality)
            # This prunes the search space for the expensive greedy algorithm.
            for c in all_cols:
                if df_sample[c].nunique() / N_sample > distinct_value_threshold:
                    bad_cols.append(c)
                else:
                    good_cols.append(c)
        else:
            return df_processed # Return empty dataframe with correct columns

        # Order bad columns by their number of unique values, descending.
        # These are least likely to contribute to LCP, so they go last.
        if bad_cols:
            bad_cols_nunique = df_sample[bad_cols].nunique()
            bad_cols = bad_cols_nunique.sort_values(ascending=False).index.tolist()
        
        # Find optimal order for good columns using the greedy conditional entropy method
        ordered_good_cols = self._find_optimal_order(df_sample[good_cols], parallel)
        
        final_order = ordered_good_cols + bad_cols
        
        return df_processed[final_order]