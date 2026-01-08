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
        # Convert all data to string to ensure consistent handling (e.g. numeric formatting)
        df_str = df.astype(str)
        
        # Apply column merges
        # Merge groups are concatenated and replace original columns
        if col_merge:
            for group in col_merge:
                if not group:
                    continue
                # Identify columns from the group that actually exist in the DataFrame
                valid_cols = [c for c in group if c in df_str.columns]
                if not valid_cols:
                    continue
                
                # New column name is concatenation of merged column names
                new_col_name = "".join(valid_cols)
                
                # If there are columns to merge, concatenate their values
                if len(valid_cols) > 0:
                    # Start with the first column
                    new_vals = df_str[valid_cols[0]].copy()
                    # Append subsequent columns
                    for c in valid_cols[1:]:
                        new_vals = new_vals + df_str[c]
                    
                    # Assign new column and drop old ones
                    df_str[new_col_name] = new_vals
                    # Only drop if the new column name is not one of the old names 
                    # (though usually new name is combo, check to be safe)
                    cols_to_drop = [c for c in valid_cols if c != new_col_name]
                    if cols_to_drop:
                        df_str.drop(columns=cols_to_drop, inplace=True)

        # Get current columns after merge
        columns = list(df_str.columns)
        n_cols = len(columns)
        n_rows = len(df_str)
        
        if n_cols == 0:
            return df_str
            
        # Precompute string lengths for all columns
        # This speeds up the gain calculation inside the loop
        df_lens = df_str.apply(lambda x: x.str.len())
        
        # Greedy search variables
        selected_cols = []
        remaining_cols = set(columns)
        
        # Group tracking:
        # group_ids: array mapping each row to a group ID based on prefix match
        # mask: boolean array indicating if a row belongs to a group with size > 1
        # (Rows in groups of size 1 are unique and don't contribute to further LCP growth)
        group_ids = np.zeros(n_rows, dtype=np.int32)
        mask = np.ones(n_rows, dtype=bool)
        
        # Iterate to select columns one by one
        for _ in range(n_cols):
            # If no rows are in non-unique groups, order of remaining columns doesn't matter
            if not mask.any():
                break
                
            # Get indices of active rows
            active_indices = np.where(mask)[0]
            if len(active_indices) == 0:
                break
            
            # Current group assignments for active rows
            current_group_ids = group_ids[active_indices]
            
            best_gain = -1.0
            best_col = None
            
            # Evaluate each remaining column
            for col in remaining_cols:
                # Get values and lengths for active rows
                vals = df_str[col].values[active_indices]
                lens = df_lens[col].values[active_indices]
                
                # Calculate Gain:
                # Gain = (Sum of lengths of all items) - (Sum of lengths of unique items per group)
                # This represents the total length of prefixes "saved" or "shared" by this column choice.
                
                total_len = np.sum(lens)
                
                # To find sum of lengths of unique items per group efficiently:
                # Create a temporary DataFrame of (group, value, length)
                tmp_df = pd.DataFrame({'g': current_group_ids, 'v': vals, 'l': lens})
                # Drop duplicates to get unique (group, value) pairs
                unique_df = tmp_df.drop_duplicates(subset=['g', 'v'])
                
                penalty = unique_df['l'].sum()
                gain = total_len - penalty
                
                if gain > best_gain:
                    best_gain = gain
                    best_col = col
            
            # If for some reason no column selected (unlikely), pick one
            if best_col is None:
                best_col = list(remaining_cols)[0]
            
            selected_cols.append(best_col)
            remaining_cols.remove(best_col)
            
            # Update groups based on the selected column
            best_vals = df_str[best_col].values[active_indices]
            
            # Create new sub-groups
            grp_df = pd.DataFrame({'g': current_group_ids, 'v': best_vals})
            # Assign new group IDs (0 to K-1)
            new_ids = grp_df.groupby(['g', 'v'], sort=False).ngroup().values
            
            # Update mask: only keep rows that are in groups with > 1 members
            if len(new_ids) > 0:
                counts = np.bincount(new_ids)
                subset_mask = counts[new_ids] > 1
                
                # Update global state
                mask[active_indices] = subset_mask
                group_ids[active_indices] = new_ids
            else:
                mask[:] = False
        
        # Append any remaining columns
        selected_cols.extend(list(remaining_cols))
        
        return df_str[selected_cols]