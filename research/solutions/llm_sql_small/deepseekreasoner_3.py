import pandas as pd
import numpy as np
from collections import defaultdict
from itertools import permutations
import time
from typing import List, Tuple, Optional
import math

class TrieNode:
    __slots__ = ('children', 'count')
    
    def __init__(self):
        self.children = {}
        self.count = 0

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
        start_time = time.time()
        
        # Apply column merges if specified
        if col_merge:
            df = self._apply_column_merges(df, col_merge)
        
        # Get column names
        columns = list(df.columns)
        n_cols = len(columns)
        
        # If only 1 column, return as is
        if n_cols <= 1:
            return df
        
        # Convert all values to strings and cache
        str_data = df.astype(str).values
        
        # Check if we need to use early stopping
        total_permutations = math.factorial(n_cols)
        if total_permutations > early_stop:
            # Use heuristic search for large number of permutations
            best_order = self._heuristic_search(str_data, columns, distinct_value_threshold)
        else:
            # Use exhaustive search for small number of permutations
            best_order = self._exhaustive_search(str_data, columns, start_time)
        
        # Reorder DataFrame columns
        return df[best_order]
    
    def _apply_column_merges(self, df: pd.DataFrame, col_merge: list) -> pd.DataFrame:
        """Apply column merges as specified."""
        df = df.copy()
        for group in col_merge:
            if len(group) <= 1:
                continue
            
            # Merge columns by concatenating their string values
            merged_values = df[group].astype(str).apply(lambda row: ''.join(row.values), axis=1)
            
            # Create new column name
            new_col_name = '_'.join(group)
            
            # Add merged column and remove original ones
            df[new_col_name] = merged_values
            df = df.drop(columns=group)
        
        return df
    
    def _calculate_hit_rate(self, data: np.ndarray, order: List[int]) -> float:
        """Calculate hit rate for a given column order."""
        n_rows = len(data)
        
        # Build concatenated strings in the given order
        strings = []
        lengths = []
        for i in range(n_rows):
            row_str = ''.join(data[i, j] for j in order)
            strings.append(row_str)
            lengths.append(len(row_str))
        
        # Calculate hit rate using trie
        total_lcp = 0
        trie = TrieNode()
        
        for i in range(n_rows):
            s = strings[i]
            node = trie
            lcp = 0
            
            for char in s:
                if char in node.children:
                    node = node.children[char]
                    lcp += 1
                else:
                    node.children[char] = TrieNode()
                    node = node.children[char]
                    break
            else:
                lcp = len(s)  # Exact match found
            
            total_lcp += lcp
            node.count += 1
        
        # Calculate total length
        total_length = sum(lengths)
        
        # Avoid division by zero
        if total_length == 0:
            return 0.0
        
        return total_lcp / total_length
    
    def _exhaustive_search(self, data: np.ndarray, columns: List[str], 
                          start_time: float) -> List[str]:
        """Exhaustive search over all permutations (for small n_cols)."""
        n_cols = len(columns)
        indices = list(range(n_cols))
        
        best_score = -1.0
        best_order = indices
        
        # Try all permutations
        for perm in permutations(indices):
            # Check timeout
            if time.time() - start_time > 8:  # Leave 2 seconds margin
                break
                
            score = self._calculate_hit_rate(data, perm)
            if score > best_score:
                best_score = score
                best_order = perm
        
        return [columns[i] for i in best_order]
    
    def _heuristic_search(self, data: np.ndarray, columns: List[str],
                         distinct_value_threshold: float) -> List[str]:
        """Heuristic search for good column order."""
        n_cols = len(columns)
        n_rows = len(data)
        
        # Calculate column statistics
        col_stats = []
        for j in range(n_cols):
            col_values = data[:, j]
            unique_values = len(set(col_values))
            distinct_ratio = unique_values / n_rows
            
            # Calculate average length
            avg_len = sum(len(val) for val in col_values) / n_rows
            
            col_stats.append({
                'index': j,
                'distinct_ratio': distinct_ratio,
                'avg_length': avg_len,
                'priority': (1 - distinct_ratio) * avg_len  # Higher priority for less distinct, longer columns
            })
        
        # Sort columns by priority (descending)
        col_stats.sort(key=lambda x: x['priority'], reverse=True)
        
        # Start with highest priority column
        best_order = [col_stats[0]['index']]
        remaining = [stat['index'] for stat in col_stats[1:]]
        
        # Greedy insertion
        while remaining:
            best_pos = -1
            best_col = -1
            best_score = -1
            
            # Try each remaining column at each position
            for col_idx in remaining:
                for pos in range(len(best_order) + 1):
                    test_order = best_order[:pos] + [col_idx] + best_order[pos:]
                    score = self._calculate_hit_rate(data, test_order)
                    
                    if score > best_score:
                        best_score = score
                        best_col = col_idx
                        best_pos = pos
            
            # Insert the best column at the best position
            best_order.insert(best_pos, best_col)
            remaining.remove(best_col)
        
        # Convert indices to column names
        return [columns[i] for i in best_order]