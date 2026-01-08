import pandas as pd
import numpy as np
from collections import defaultdict, Counter
from typing import List, Tuple, Dict, Set
import itertools
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
        # Apply column merges if specified
        if col_merge:
            df = self._merge_columns(df, col_merge)
        
        # Get the column order that maximizes prefix hit rate
        best_order = self._optimize_column_order(
            df, 
            early_stop,
            row_stop,
            col_stop,
            distinct_value_threshold,
            parallel
        )
        
        # Return dataframe with optimized column order
        return df[best_order]
    
    def _merge_columns(self, df: pd.DataFrame, col_merge: List[List[str]]) -> pd.DataFrame:
        """Merge columns according to the specification."""
        merged_df = df.copy()
        columns_to_drop = []
        
        for group in col_merge:
            if all(col in merged_df.columns for col in group):
                # Create merged column name
                merged_name = "_".join(group)
                # Concatenate column values as strings
                merged_df[merged_name] = merged_df[group].astype(str).agg(''.join, axis=1)
                columns_to_drop.extend(group)
        
        # Drop original columns that were merged
        merged_df = merged_df.drop(columns=columns_to_drop)
        return merged_df
    
    def _optimize_column_order(
        self,
        df: pd.DataFrame,
        early_stop: int,
        row_stop: int,
        col_stop: int,
        distinct_value_threshold: float,
        parallel: bool
    ) -> List[str]:
        """Optimize column order to maximize prefix hit rate."""
        n_rows, n_cols = df.shape
        
        # Calculate column statistics
        col_stats = {}
        for col in df.columns:
            col_series = df[col].astype(str)
            # Distinct value ratio
            distinct_ratio = col_series.nunique() / n_rows
            # Average string length
            avg_len = col_series.str.len().mean()
            # Value frequency for top values
            value_counts = col_series.value_counts()
            col_stats[col] = {
                'distinct_ratio': distinct_ratio,
                'avg_len': avg_len,
                'top_values': value_counts.iloc[:5].to_dict(),
                'value_counts': value_counts
            }
        
        # Group columns by distinctiveness
        low_distinct_cols = []
        high_distinct_cols = []
        
        for col, stats in col_stats.items():
            if stats['distinct_ratio'] < distinct_value_threshold:
                low_distinct_cols.append(col)
            else:
                high_distinct_cols.append(col)
        
        # Sort low distinct columns by average length (longer first for better prefix matching)
        low_distinct_cols.sort(key=lambda x: col_stats[x]['avg_len'], reverse=True)
        
        # Sort high distinct columns by distinct ratio (lower first)
        high_distinct_cols.sort(key=lambda x: col_stats[x]['distinct_ratio'])
        
        # Start with low distinct columns
        current_order = low_distinct_cols + high_distinct_cols
        
        # Evaluate and refine the order
        best_order = current_order.copy()
        best_score = self._evaluate_order(df, best_order)
        
        # Try local optimizations
        for _ in range(min(3, n_cols)):  # Limited iterations due to runtime constraint
            improved = False
            
            # Try swapping adjacent columns
            for i in range(len(current_order) - 1):
                new_order = current_order.copy()
                new_order[i], new_order[i + 1] = new_order[i + 1], new_order[i]
                new_score = self._evaluate_order(df, new_order)
                
                if new_score > best_score:
                    best_score = new_score
                    best_order = new_order.copy()
                    improved = True
                    break
            
            if not improved:
                break
            
            current_order = best_order.copy()
        
        # Try moving columns with high prefix potential to front
        prefix_potential = {}
        for col in df.columns:
            # Estimate prefix potential based on early value matches
            col_values = df[col].astype(str).values
            prefix_len = 0
            for i in range(1, min(100, n_rows)):
                prefix_len += self._lcp_length(col_values[i], col_values[i-1])
            prefix_potential[col] = prefix_len / min(100, n_rows)
        
        # Create alternative order based on prefix potential
        potential_order = sorted(df.columns, key=lambda x: prefix_potential[x], reverse=True)
        potential_score = self._evaluate_order(df, potential_order)
        
        if potential_score > best_score:
            best_order = potential_order
            best_score = potential_score
        
        return best_order
    
    def _evaluate_order(self, df: pd.DataFrame, column_order: List[str]) -> float:
        """Evaluate the hit rate for a given column order."""
        n_rows = len(df)
        if n_rows < 2:
            return 0.0
        
        # Create concatenated strings in order
        strings = []
        total_len = 0
        
        # Use numpy for efficient string concatenation
        for i in range(n_rows):
            row_str = ''.join(df[col].iloc[i].astype(str) for col in column_order)
            strings.append(row_str)
            total_len += len(row_str)
        
        # Calculate hit rate
        total_lcp = 0
        seen_strings = [strings[0]]
        
        for i in range(1, n_rows):
            max_lcp = 0
            current_str = strings[i]
            
            # Check against previous strings
            for prev_str in seen_strings[-min(100, i):]:  # Limit check to recent strings
                lcp_len = self._lcp_length(current_str, prev_str)
                if lcp_len > max_lcp:
                    max_lcp = lcp_len
            
            total_lcp += max_lcp
            seen_strings.append(current_str)
        
        return total_lcp / total_len if total_len > 0 else 0.0
    
    def _lcp_length(self, s1: str, s2: str) -> int:
        """Calculate length of longest common prefix between two strings."""
        min_len = min(len(s1), len(s2))
        for i in range(min_len):
            if s1[i] != s2[i]:
                return i
        return min_len