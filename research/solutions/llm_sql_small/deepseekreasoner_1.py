import pandas as pd
import numpy as np
from itertools import permutations
from collections import defaultdict
import time

class TrieNode:
    __slots__ = ('children', 'count')
    def __init__(self):
        self.children = {}
        self.count = 0

class Trie:
    def __init__(self):
        self.root = TrieNode()
    
    def insert(self, word: str):
        node = self.root
        for ch in word:
            if ch not in node.children:
                node.children[ch] = TrieNode()
            node = node.children[ch]
            node.count += 1
    
    def longest_common_prefix(self, word: str) -> int:
        node = self.root
        length = 0
        for ch in word:
            if ch in node.children:
                node = node.children[ch]
                length += 1
            else:
                break
        return length

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
        # Start timing
        start_time = time.time()
        
        # Apply column merges if specified
        if col_merge:
            df = self._apply_merges(df, col_merge)
        
        # Convert all values to strings and precompute column data
        str_df = df.astype(str)
        n_rows, n_cols = str_df.shape
        
        # If number of columns is small, try exact search with pruning
        if n_cols <= 6:
            best_order, _ = self._exact_search(str_df)
            return df[best_order]
        
        # For larger column counts, use heuristic approach
        # 1. First try greedy approach based on column distinctiveness and prefix potential
        greedy_order = self._greedy_ordering(str_df)
        
        # 2. Refine with local search (swap-based optimization)
        best_order = self._local_search(str_df, greedy_order, start_time)
        
        return df[best_order]
    
    def _apply_merges(self, df: pd.DataFrame, col_merge: list) -> pd.DataFrame:
        """Apply column merges by concatenating specified columns."""
        result_df = df.copy()
        
        for merge_group in col_merge:
            if len(merge_group) < 2:
                continue
                
            # Use the first column name as the merged column name
            main_col = merge_group[0]
            other_cols = merge_group[1:]
            
            # Concatenate values without spaces
            def merge_row(row):
                parts = [str(row[main_col])]
                for col in other_cols:
                    parts.append(str(row[col]))
                return ''.join(parts)
            
            result_df[main_col] = result_df.apply(merge_row, axis=1)
            
            # Drop the merged columns
            result_df = result_df.drop(columns=other_cols)
        
        return result_df
    
    def _compute_prefix_score(self, df: pd.DataFrame, col_order: list) -> float:
        """Compute the prefix hit rate score for a given column order."""
        n_rows = len(df)
        trie = Trie()
        total_lcp = 0
        total_len = 0
        
        for i in range(n_rows):
            # Build concatenated string for this row in the given column order
            row_str = ''.join(df.iloc[i][col] for col in col_order)
            total_len += len(row_str)
            
            if i > 0:  # Skip first row for LCP calculation
                lcp = trie.longest_common_prefix(row_str)
                total_lcp += lcp
            
            trie.insert(row_str)
        
        return total_lcp / total_len if total_len > 0 else 0.0
    
    def _exact_search(self, df: pd.DataFrame):
        """Exhaustive search for small number of columns (≤6)."""
        cols = list(df.columns)
        n_cols = len(cols)
        best_score = -1
        best_order = cols
        
        # Try all permutations
        for perm in permutations(cols):
            perm_list = list(perm)
            score = self._compute_prefix_score(df, perm_list)
            if score > best_score:
                best_score = score
                best_order = perm_list
        
        return best_order, best_score
    
    def _greedy_ordering(self, df: pd.DataFrame) -> list:
        """Greedy heuristic for column ordering."""
        cols = list(df.columns)
        n_cols = len(cols)
        
        # Precompute column statistics
        col_stats = []
        for col in cols:
            values = df[col].tolist()
            # Estimate distinctiveness and prefix potential
            distinct_ratio = len(set(values)) / len(values)
            avg_len = np.mean([len(str(v)) for v in values])
            
            # Score: lower distinctiveness and longer strings are better for early positions
            score = (1 - distinct_ratio) * avg_len
            col_stats.append((col, score, distinct_ratio, avg_len))
        
        # Sort by score descending (best columns first)
        col_stats.sort(key=lambda x: x[1], reverse=True)
        
        # Start with best column and greedily add next best
        ordered = [col_stats[0][0]]
        remaining = [col for col, _, _, _ in col_stats[1:]]
        
        while remaining:
            best_next = None
            best_score = -1
            
            # Try each remaining column in next position
            for i, next_col in enumerate(remaining):
                temp_order = ordered + [next_col]
                score = self._compute_prefix_score(df, temp_order)
                
                if score > best_score:
                    best_score = score
                    best_next = next_col
                    best_idx = i
            
            if best_next:
                ordered.append(best_next)
                remaining.pop(best_idx)
        
        return ordered
    
    def _local_search(self, df: pd.DataFrame, initial_order: list, start_time: float) -> list:
        """Local search optimization starting from initial order."""
        current_order = initial_order.copy()
        current_score = self._compute_prefix_score(df, current_order)
        n_cols = len(current_order)
        improved = True
        
        # Limit search iterations to avoid timeout
        max_iterations = 100
        iteration = 0
        
        while improved and iteration < max_iterations:
            improved = False
            
            # Try all pairwise swaps
            for i in range(n_cols):
                for j in range(i + 1, n_cols):
                    # Check time limit
                    if time.time() - start_time > 9.5:  # Leave some margin
                        return current_order
                    
                    new_order = current_order.copy()
                    new_order[i], new_order[j] = new_order[j], new_order[i]
                    
                    new_score = self._compute_prefix_score(df, new_order)
                    
                    if new_score > current_score:
                        current_order = new_order
                        current_score = new_score
                        improved = True
                        break  # Restart from beginning after improvement
                
                if improved:
                    break
            
            iteration += 1
        
        return current_order