import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
import sys
import time

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
        # Apply column merges if specified
        if col_merge is not None:
            df = self._apply_column_merges(df, col_merge)
        
        # If only one column, return as is
        if len(df.columns) <= 1:
            return df
        
        # Preprocess column data
        col_data, initial_scores = self._preprocess_columns(df)
        
        # Greedy column ordering
        permutation = self._greedy_column_ordering(
            df, col_data, initial_scores, 
            early_stop, row_stop, col_stop, parallel
        )
        
        # Return DataFrame with reordered columns
        return df[permutation]
    
    def _apply_column_merges(self, df: pd.DataFrame, col_merge: list) -> pd.DataFrame:
        """Merge columns according to col_merge specification."""
        df = df.copy()
        for i, group in enumerate(col_merge):
            # Check all columns exist
            existing_cols = [col for col in group if col in df.columns]
            if not existing_cols:
                continue
            
            # Concatenate values as strings
            merged_values = df[existing_cols].astype(str).agg(''.join, axis=1)
            new_name = f'__merged_{i}'
            df[new_name] = merged_values
            df.drop(columns=existing_cols, inplace=True)
        return df
    
    def _preprocess_columns(self, df: pd.DataFrame):
        """Precompute column data and initial scores."""
        col_data = {}
        initial_scores = {}
        
        for col in df.columns:
            # Convert to strings
            str_series = df[col].astype(str)
            values = str_series.tolist()
            
            # Create mapping from value to integer ID
            unique_vals = list(set(values))
            val_to_id = {v: i for i, v in enumerate(unique_vals)}
            lengths = [len(v) for v in unique_vals]
            
            # Convert to numpy arrays for efficiency
            value_ids = np.array([val_to_id[v] for v in values], dtype=np.int32)
            lengths_arr = np.array(lengths, dtype=np.int32)
            
            col_data[col] = {
                'value_ids': value_ids,
                'lengths': lengths_arr,
            }
            
            # Compute initial score: increase if placed first
            counts = np.bincount(value_ids)
            increase = np.sum((counts - 1) * lengths_arr)
            initial_scores[col] = increase
        
        return col_data, initial_scores
    
    def _compute_increase(
        self, 
        col: str, 
        col_info: Dict[str, np.ndarray], 
        current_groups: List[List[int]], 
        row_stop: int
    ) -> float:
        """Compute increase in objective from adding this column."""
        value_ids = col_info['value_ids']
        lengths = col_info['lengths']
        total_increase = 0
        
        for group in current_groups:
            if len(group) <= row_stop:
                continue
            
            # Get value IDs for rows in this group
            ids = value_ids[group]
            
            # Count frequencies
            counts = np.bincount(ids)
            
            # Add contribution from values with count > 1
            for val_id, cnt in enumerate(counts):
                if cnt > 1:
                    total_increase += (cnt - 1) * lengths[val_id]
        
        return total_increase
    
    def _greedy_column_ordering(
        self,
        df: pd.DataFrame,
        col_data: Dict[str, Any],
        initial_scores: Dict[str, float],
        early_stop: int,
        row_stop: int,
        col_stop: int,
        parallel: bool
    ) -> List[str]:
        """Greedy algorithm to select column order."""
        columns = list(df.columns)
        N = len(df)
        permutation = []
        remaining_cols = columns.copy()
        
        # Initial groups: all rows together
        current_groups = [list(range(N))]
        
        # Try to use parallel processing if requested
        use_parallel = parallel
        try:
            from joblib import Parallel, delayed
        except ImportError:
            use_parallel = False
        
        while remaining_cols:
            # Select candidates to evaluate
            if col_stop > 0 and len(remaining_cols) > col_stop:
                candidates = sorted(
                    remaining_cols, 
                    key=lambda c: initial_scores[c], 
                    reverse=True
                )[:col_stop]
            else:
                candidates = remaining_cols
            
            # Evaluate candidates
            best_col = None
            best_increase = -1
            
            if use_parallel and len(candidates) > 1:
                # Parallel evaluation
                increases = Parallel(n_jobs=-1)(
                    delayed(self._compute_increase)(
                        c, col_data[c], current_groups, row_stop
                    ) for c in candidates
                )
                for c, inc in zip(candidates, increases):
                    if inc > best_increase:
                        best_increase = inc
                        best_col = c
            else:
                # Serial evaluation
                for c in candidates:
                    inc = self._compute_increase(
                        c, col_data[c], current_groups, row_stop
                    )
                    if inc > best_increase:
                        best_increase = inc
                        best_col = c
            
            # Early stopping condition
            if best_increase <= early_stop and len(permutation) > 0:
                # Add best column first
                permutation.append(best_col)
                remaining_cols.remove(best_col)
                # Add remaining columns by initial score
                remaining_cols_sorted = sorted(
                    remaining_cols, 
                    key=lambda c: initial_scores[c], 
                    reverse=True
                )
                permutation.extend(remaining_cols_sorted)
                break
            
            # Add best column to permutation
            permutation.append(best_col)
            remaining_cols.remove(best_col)
            
            # Update groups by splitting with the new column
            new_groups = []
            value_ids = col_data[best_col]['value_ids']
            
            for group in current_groups:
                # Split group by column value
                subgroups = {}
                for row_idx in group:
                    vid = value_ids[row_idx]
                    subgroups.setdefault(vid, []).append(row_idx)
                
                # Add subgroups to new groups
                for subgroup in subgroups.values():
                    new_groups.append(subgroup)
            
            current_groups = new_groups
        
        return permutation