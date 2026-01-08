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
        # Convert all data to string to ensure consistent processing
        df_str = df.astype(str)
        
        # 1. Apply column merges
        if col_merge:
            for group in col_merge:
                # Identify valid columns in this group that exist in the DataFrame
                valid_cols = [c for c in group if c in df_str.columns]
                if not valid_cols:
                    continue
                
                # New column name: concatenation of names (using underscore for uniqueness)
                new_col_name = "_".join(valid_cols)
                
                # Perform string concatenation of the column values
                merged_series = df_str[valid_cols[0]]
                for c in valid_cols[1:]:
                    merged_series = merged_series + df_str[c]
                
                # Update DataFrame: add merged column and drop originals
                df_str[new_col_name] = merged_series
                df_str.drop(columns=valid_cols, inplace=True)
        
        # 2. Precompute statistics and codes for remaining columns
        # We transform each column into integer codes for fast grouping operations.
        cols = list(df_str.columns)
        col_data = {}
        
        for c in cols:
            # pd.factorize returns integer codes (0..K-1) and uniques
            codes, uniques = pd.factorize(df_str[c], sort=False)
            # Calculate average string length for tie-breaking
            avg_len = df_str[c].str.len().mean()
            
            col_data[c] = {
                'codes': codes,
                'cardinality': len(uniques),
                'avg_len': avg_len
            }
            
        # 3. Greedy optimization
        # Goal: Order columns to minimize the growth of unique prefixes (maximize sharing).
        # We maintain `current_groups`, an array of IDs representing the unique prefix of each row.
        
        N = len(df_str)
        current_groups = np.zeros(N, dtype=np.int64)
        current_nunique = 1
        
        selected_cols = []
        remaining_cols = set(cols)
        
        while remaining_cols:
            # Optimization: If all rows are already fully distinguished (N unique prefixes),
            # the order of remaining columns has minimal impact on the hit rate numerator.
            # We append the rest sorted by length (descending) as a heuristic to maximize
            # potential overlap length if any accidental matches occur.
            if current_nunique == N:
                sorted_rest = sorted(list(remaining_cols), key=lambda c: col_data[c]['avg_len'], reverse=True)
                selected_cols.extend(sorted_rest)
                break
                
            best_col = None
            # Metric: (num_unique_groups, -avg_length) 
            # We want to minimize num_unique_groups. If tied, maximize avg_length.
            best_metric = (float('inf'), float('-inf'))
            
            # Special case for the first column (no combination needed)
            if not selected_cols:
                # Pick column with lowest cardinality, tie-break with max length
                candidates = list(remaining_cols)
                candidates.sort(key=lambda c: (col_data[c]['cardinality'], -col_data[c]['avg_len']))
                best_col = candidates[0]
                
                # Update state
                selected_cols.append(best_col)
                remaining_cols.remove(best_col)
                current_groups = col_data[best_col]['codes']
                current_nunique = col_data[best_col]['cardinality']
                continue
                
            # Search for best next column
            candidates = list(remaining_cols)
            
            for col in candidates:
                c_codes = col_data[col]['codes']
                c_card = col_data[col]['cardinality']
                
                # Combine current group IDs with candidate column codes
                # Mapping: new_id = current_group * card + code
                # Using int64 to avoid overflow (max val approx N^2)
                combined = current_groups * np.int64(c_card) + c_codes.astype(np.int64)
                
                # Calculate resulting number of unique groups using numpy
                n_uniq = len(np.unique(combined))
                
                metric = (n_uniq, -col_data[col]['avg_len'])
                
                if metric < best_metric:
                    best_metric = metric
                    best_col = col
            
            # Select best column
            selected_cols.append(best_col)
            remaining_cols.remove(best_col)
            
            # Update current_groups for next iteration
            c_codes = col_data[best_col]['codes']
            c_card = col_data[best_col]['cardinality']
            combined = current_groups * np.int64(c_card) + c_codes.astype(np.int64)
            
            # Re-factorize to keep group IDs compact (0..U-1) for the next multiplication step
            current_groups, _ = pd.factorize(combined, sort=False)
            current_nunique = best_metric[0]
            
        return df_str[selected_cols]