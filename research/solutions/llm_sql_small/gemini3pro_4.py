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
        Implements column merging and a beam search strategy to optimize the LCP metric.
        """
        df_curr = df.copy()
        
        # 1. Apply Column Merges
        if col_merge:
            for group in col_merge:
                # Filter group cols that exist in the dataframe
                valid_group = [c for c in group if c in df_curr.columns]
                if not valid_group:
                    continue
                
                # Concatenate values of columns in the group
                merged_series = df_curr[valid_group[0]].astype(str)
                for col in valid_group[1:]:
                    merged_series = merged_series + df_curr[col].astype(str)
                
                # Generate new column name (concatenation of original names)
                new_col_name = "".join(valid_group)
                
                # Drop original columns and insert the merged one
                df_curr.drop(columns=valid_group, inplace=True)
                df_curr[new_col_name] = merged_series
        
        cols = df_curr.columns.tolist()
        n_rows = len(df_curr)
        
        # Edge cases
        if n_rows == 0 or len(cols) <= 1:
            return df_curr
            
        # 2. Precompute column statistics
        # We need average length and integer codes for each column to speed up the search
        col_stats = {}
        for c in cols:
            s = df_curr[c].astype(str)
            codes, uniques = pd.factorize(s, sort=False)
            col_stats[c] = {
                'len': s.str.len().mean(),
                'codes': codes,
                'max_code': len(uniques)
            }
            
        # 3. Beam Search
        # Objective: Maximize sum of L_c * (N - U_prefix)
        # This proxy strictly correlates with the problem objective of maximizing total LCP.
        # State: (score, group_ids_array, selected_cols_tuple)
        
        # Initial state: all rows in the same group (group 0), score 0
        initial_groups = np.zeros(n_rows, dtype=np.int64)
        beam = [(0.0, initial_groups, ())]
        
        # Beam width parameter: Balances runtime vs quality.
        # Given 10s constraint and ~28k rows, width 4 is safe and effective.
        beam_width = 4
        
        for step in range(len(cols)):
            candidates = []
            for score, group_ids, selected in beam:
                selected_set = set(selected)
                remaining = [c for c in cols if c not in selected_set]
                
                for c in remaining:
                    stats = col_stats[c]
                    c_codes = stats['codes']
                    c_len = stats['len']
                    c_max = stats['max_code']
                    
                    # Calculate new group IDs by combining current groups with the new column
                    # We map pairs (group_id, c_code) to new unique integers.
                    # Multiplier ensures uniqueness of the combination.
                    multiplier = c_max + 1
                    packed = group_ids * multiplier + c_codes
                    
                    # Factorize to get normalized group IDs (0..U-1) and count U
                    new_ids, new_uniques = pd.factorize(packed, sort=False)
                    u_count = len(new_uniques)
                    
                    # Calculate marginal gain
                    # Gain = Average Length * Number of rows that share prefix with at least one previous row
                    # Count of such rows = Total Rows - Number of Unique Prefixes
                    term = c_len * (n_rows - u_count)
                    new_score = score + term
                    
                    candidates.append((new_score, new_ids, selected + (c,)))
            
            if not candidates:
                break
                
            # Select top K candidates based on score
            candidates.sort(key=lambda x: x[0], reverse=True)
            beam = candidates[:beam_width]
            
        # Extract best permutation
        best_perm = beam[0][2]
        
        # Return DataFrame with reordered columns
        return df_curr[list(best_perm)]