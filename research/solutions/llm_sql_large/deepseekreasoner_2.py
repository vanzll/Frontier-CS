import pandas as pd
import numpy as np
from collections import defaultdict
from typing import List, Tuple
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
            df = self._apply_column_merges(df, col_merge)
        
        # If only 1 column left, return as is
        if len(df.columns) <= 1:
            return df
        
        # Convert all values to strings and cache lengths
        str_df = df.astype(str)
        col_lengths = {col: str_df[col].str.len().mean() for col in df.columns}
        
        # Analyze column properties
        col_stats = self._analyze_columns(str_df, distinct_value_threshold)
        
        # Determine initial column ordering
        ordered_cols = self._get_initial_ordering(str_df, col_stats, col_lengths)
        
        # Optimize ordering with beam search
        optimized_order = self._optimize_order(
            str_df, ordered_cols, col_lengths, 
            early_stop, row_stop, col_stop
        )
        
        # Return reordered DataFrame
        return df[optimized_order]
    
    def _apply_column_merges(self, df: pd.DataFrame, col_merge: list) -> pd.DataFrame:
        """Merge columns as specified in col_merge."""
        result_df = df.copy()
        
        for merge_group in col_merge:
            if len(merge_group) < 2:
                continue
                
            # Use the first column name as the merged column name
            main_col = merge_group[0]
            other_cols = merge_group[1:]
            
            # Merge columns by concatenating string values
            result_df[main_col] = result_df[merge_group].astype(str).apply(
                lambda row: ''.join(row), axis=1
            )
            
            # Remove the other merged columns
            result_df = result_df.drop(columns=other_cols)
        
        return result_df
    
    def _analyze_columns(self, df: pd.DataFrame, threshold: float) -> dict:
        """Analyze column statistics for ordering decisions."""
        n_rows = len(df)
        stats = {}
        
        for col in df.columns:
            values = df[col]
            unique_count = values.nunique()
            stats[col] = {
                'unique_ratio': unique_count / n_rows,
                'is_almost_unique': (unique_count / n_rows) > threshold,
                'value_lengths': values.str.len().mean()
            }
        
        return stats
    
    def _get_initial_ordering(self, df: pd.DataFrame, col_stats: dict, 
                             col_lengths: dict) -> List[str]:
        """Get initial column ordering based on heuristics."""
        cols = list(df.columns)
        n_cols = len(cols)
        
        # Score columns based on multiple factors
        scores = {}
        for col in cols:
            stats = col_stats[col]
            
            # Columns with fewer unique values (higher repetition) should come earlier
            uniqueness_penalty = stats['unique_ratio']
            
            # Longer columns should come later (they're less likely to match)
            length_penalty = col_lengths[col] / max(col_lengths.values()) if max(col_lengths.values()) > 0 else 1
            
            # Combine factors
            score = uniqueness_penalty * 0.7 + length_penalty * 0.3
            scores[col] = score
        
        # Sort by score (lower is better for earlier positions)
        sorted_cols = sorted(cols, key=lambda x: scores[x])
        
        return sorted_cols
    
    def _compute_prefix_score(self, df: pd.DataFrame, order: List[str], 
                             sample_size: int = None) -> float:
        """Compute the prefix hit rate for a given column order."""
        if sample_size and sample_size < len(df):
            sample_df = df.sample(n=min(sample_size, len(df)), random_state=42)
        else:
            sample_df = df
        
        # Create concatenated strings in the given order
        strings = []
        for _, row in sample_df[order].iterrows():
            strings.append(''.join(row.values.astype(str)))
        
        # Compute prefix hit rate
        total_lcp = 0
        total_length = 0
        
        # Store prefixes in a trie-like structure for efficiency
        prefix_dict = defaultdict(set)
        
        for i, s in enumerate(strings):
            total_length += len(s)
            
            if i == 0:
                # Add first string's prefixes
                for j in range(1, len(s) + 1):
                    prefix_dict[s[:j]].add(i)
                continue
            
            # Find longest common prefix with any previous string
            max_lcp = 0
            # Check progressively shorter prefixes
            for prefix_len in range(min(len(s), 100), 0, -1):
                prefix = s[:prefix_len]
                if prefix in prefix_dict and prefix_dict[prefix]:
                    max_lcp = prefix_len
                    break
            
            total_lcp += max_lcp
            
            # Add current string's prefixes to dictionary
            for j in range(1, len(s) + 1):
                prefix_dict[s[:j]].add(i)
        
        if total_length == 0:
            return 0.0
        
        return total_lcp / total_length
    
    def _optimize_order(self, df: pd.DataFrame, initial_order: List[str],
                       col_lengths: dict, early_stop: int, 
                       row_stop: int, col_stop: int) -> List[str]:
        """Optimize column order using beam search with local improvements."""
        n_cols = len(initial_order)
        if n_cols <= 2:
            return initial_order
        
        # Use smaller sample for faster evaluation during optimization
        sample_size = min(row_stop * 100, len(df))
        if sample_size < 100:
            sample_size = min(100, len(df))
        
        current_order = initial_order.copy()
        best_order = current_order.copy()
        best_score = self._compute_prefix_score(df, current_order, sample_size)
        
        # Beam search with limited width
        beam_width = min(col_stop * 2, n_cols // 2 + 1)
        beam_width = max(2, beam_width)
        
        # Track visited states to avoid cycles
        visited = set()
        visited.add(tuple(current_order))
        
        # Local optimization: try swapping adjacent columns
        improvement = True
        iterations = 0
        max_iterations = min(early_stop // 100, 100)
        
        while improvement and iterations < max_iterations:
            improvement = False
            
            # Try all adjacent swaps
            for i in range(n_cols - 1):
                new_order = current_order.copy()
                new_order[i], new_order[i + 1] = new_order[i + 1], new_order[i]
                
                order_tuple = tuple(new_order)
                if order_tuple in visited:
                    continue
                
                score = self._compute_prefix_score(df, new_order, sample_size)
                visited.add(order_tuple)
                
                if score > best_score:
                    best_score = score
                    best_order = new_order.copy()
                    current_order = new_order.copy()
                    improvement = True
                    break
            
            if not improvement:
                # Try swapping non-adjacent columns that are far apart
                for i in range(n_cols):
                    for j in range(i + 2, min(i + beam_width, n_cols)):
                        new_order = current_order.copy()
                        new_order[i], new_order[j] = new_order[j], new_order[i]
                        
                        order_tuple = tuple(new_order)
                        if order_tuple in visited:
                            continue
                        
                        score = self._compute_prefix_score(df, new_order, sample_size)
                        visited.add(order_tuple)
                        
                        if score > best_score:
                            best_score = score
                            best_order = new_order.copy()
                            current_order = new_order.copy()
                            improvement = True
                            break
                    
                    if improvement:
                        break
            
            iterations += 1
        
        # Final verification with slightly larger sample
        final_sample = min(sample_size * 2, len(df))
        final_score = self._compute_prefix_score(df, best_order, final_sample)
        
        # If we didn't improve, try a different starting point
        if final_score <= best_score and n_cols > 3:
            # Try reversing the order as another starting point
            reversed_order = list(reversed(initial_order))
            if tuple(reversed_order) not in visited:
                rev_score = self._compute_prefix_score(df, reversed_order, sample_size)
                if rev_score > final_score:
                    best_order = reversed_order
                    final_score = rev_score
        
        return best_order