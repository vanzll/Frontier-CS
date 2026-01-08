import pandas as pd
import numpy as np
from itertools import permutations, combinations
from collections import defaultdict, Counter
import time
import multiprocessing as mp
from functools import lru_cache
import math

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
        # Apply column merges
        if col_merge is not None:
            df = self._apply_merges(df, col_merge)
        
        # If only 1 column, return as is
        if len(df.columns) <= 1:
            return df
        
        # Get string representation of rows
        str_rows = self._get_string_rows(df)
        
        # Analyze column properties
        col_stats = self._analyze_columns(df, str_rows, distinct_value_threshold)
        
        # Determine search strategy based on number of columns
        n_cols = len(df.columns)
        
        if n_cols <= 6:
            # Exhaustive search for small number of columns
            best_order = self._exhaustive_search(df, str_rows, col_stats, early_stop)
        elif n_cols <= 8:
            # Beam search with heuristic
            best_order = self._beam_search(df, str_rows, col_stats, early_stop, row_stop, col_stop)
        else:
            # Greedy + local search for larger columns
            best_order = self._greedy_search(df, str_rows, col_stats, early_stop, parallel)
        
        # Return reordered dataframe
        return df[best_order]
    
    def _apply_merges(self, df: pd.DataFrame, col_merge: list) -> pd.DataFrame:
        """Merge specified columns into single columns."""
        df = df.copy()
        for group in col_merge:
            if len(group) > 1:
                # Merge columns in group
                merged_name = '_'.join(group)
                df[merged_name] = df[group].astype(str).agg(''.join, axis=1)
                df = df.drop(columns=group)
        return df
    
    def _get_string_rows(self, df: pd.DataFrame):
        """Convert each row to string representation for each column."""
        return [df.iloc[i].astype(str).tolist() for i in range(len(df))]
    
    def _analyze_columns(self, df, str_rows, distinct_value_threshold):
        """Analyze column properties to guide search."""
        n_rows = len(df)
        n_cols = len(df.columns)
        
        # Calculate distinct value ratios
        distinct_ratios = []
        for col_idx in range(n_cols):
            col_values = [str_rows[i][col_idx] for i in range(n_rows)]
            distinct_count = len(set(col_values))
            distinct_ratios.append(distinct_count / n_rows)
        
        # Calculate prefix frequencies for pairs of columns
        prefix_scores = np.zeros((n_cols, n_cols))
        for i in range(n_cols):
            for j in range(n_cols):
                if i != j:
                    # Count how many rows have same value in both columns
                    same_count = sum(1 for row in range(n_rows) 
                                   if str_rows[row][i] == str_rows[row][j])
                    prefix_scores[i, j] = same_count / n_rows
        
        # Calculate column dependencies
        col_deps = []
        for i in range(n_cols):
            dep_score = 0
            for j in range(n_cols):
                if i != j:
                    dep_score += prefix_scores[i, j]
            col_deps.append(dep_score / (n_cols - 1) if n_cols > 1 else 0)
        
        return {
            'distinct_ratios': distinct_ratios,
            'prefix_scores': prefix_scores,
            'col_deps': col_deps,
            'n_rows': n_rows,
            'n_cols': n_cols
        }
    
    def _exhaustive_search(self, df, str_rows, col_stats, early_stop):
        """Exhaustive search for small number of columns."""
        n_cols = col_stats['n_cols']
        columns = list(df.columns)
        
        best_order = columns
        best_score = self._calculate_hit_rate(str_rows, list(range(n_cols)))
        
        # Generate all permutations
        all_perms = list(permutations(range(n_cols)))
        if len(all_perms) > early_stop:
            # Sample permutations if too many
            import random
            random.seed(42)
            all_perms = random.sample(all_perms, min(early_stop, len(all_perms)))
        
        for perm in all_perms:
            score = self._calculate_hit_rate(str_rows, perm)
            if score > best_score:
                best_score = score
                best_order = [columns[i] for i in perm]
        
        return best_order
    
    def _beam_search(self, df, str_rows, col_stats, early_stop, row_stop, col_stop):
        """Beam search with heuristic pruning."""
        n_cols = col_stats['n_cols']
        columns = list(df.columns)
        
        # Start with empty permutation
        beam = [([], 0.0)]
        beam_width = min(early_stop // 10, 100)
        
        for step in range(n_cols):
            new_beam = []
            for prefix, prefix_score in beam:
                remaining = [i for i in range(n_cols) if i not in prefix]
                
                # Evaluate candidates
                candidates = []
                for col in remaining:
                    new_perm = prefix + [col]
                    
                    # Estimate score using heuristic
                    if len(new_perm) <= col_stop:
                        # Calculate exact score for first few columns
                        sample_rows = min(row_stop * 1000, col_stats['n_rows'])
                        score = self._calculate_hit_rate_sample(str_rows, new_perm, sample_rows, col_stats['n_cols'])
                    else:
                        # Use heuristic estimation
                        score = prefix_score + self._estimate_column_gain(col, prefix, col_stats)
                    
                    candidates.append((new_perm, score))
                
                # Keep top candidates
                candidates.sort(key=lambda x: x[1], reverse=True)
                new_beam.extend(candidates[:beam_width])
            
            # Update beam
            new_beam.sort(key=lambda x: x[1], reverse=True)
            beam = new_beam[:beam_width]
        
        # Evaluate full permutations from beam
        best_order = columns
        best_score = 0
        
        for perm_idx, _ in beam[:min(10, len(beam))]:
            perm = [columns[i] for i in perm_idx]
            score = self._calculate_hit_rate(str_rows, perm_idx)
            if score > best_score:
                best_score = score
                best_order = perm
        
        return best_order
    
    def _greedy_search(self, df, str_rows, col_stats, early_stop, parallel):
        """Greedy search with local optimization."""
        n_cols = col_stats['n_cols']
        columns = list(df.columns)
        
        # Start with heuristic ordering
        order = self._get_initial_order(col_stats, columns)
        
        # Local optimization
        improved = True
        iteration = 0
        
        while improved and iteration < 10:
            improved = False
            iteration += 1
            
            # Try swapping adjacent pairs
            for i in range(n_cols - 1):
                new_order = order.copy()
                new_order[i], new_order[i + 1] = new_order[i + 1], new_order[i]
                
                old_idx = [columns.index(c) for c in order]
                new_idx = [columns.index(c) for c in new_order]
                
                old_score = self._calculate_hit_rate(str_rows, old_idx)
                new_score = self._calculate_hit_rate(str_rows, new_idx)
                
                if new_score > old_score:
                    order = new_order
                    improved = True
            
            # Try moving each column to different positions
            if not improved and n_cols <= 8:
                for i in range(n_cols):
                    for j in range(n_cols):
                        if i != j:
                            new_order = order.copy()
                            col = new_order.pop(i)
                            new_order.insert(j, col)
                            
                            old_idx = [columns.index(c) for c in order]
                            new_idx = [columns.index(c) for c in new_order]
                            
                            old_score = self._calculate_hit_rate(str_rows, old_idx)
                            new_score = self._calculate_hit_rate(str_rows, new_idx)
                            
                            if new_score > old_score:
                                order = new_order
                                improved = True
        
        return order
    
    def _get_initial_order(self, col_stats, columns):
        """Get initial column order based on heuristics."""
        n_cols = col_stats['n_cols']
        
        # Sort columns by distinct ratio (lowest first for prefix)
        col_indices = list(range(n_cols))
        col_indices.sort(key=lambda i: col_stats['distinct_ratios'][i])
        
        # Group columns with high dependency together
        groups = []
        used = set()
        
        for i in range(n_cols):
            if i not in used:
                group = [i]
                used.add(i)
                
                # Find columns highly dependent with this one
                for j in range(n_cols):
                    if j not in used and col_stats['prefix_scores'][i, j] > 0.3:
                        group.append(j)
                        used.add(j)
                
                groups.append(group)
        
        # Flatten groups
        order_indices = []
        for group in groups:
            order_indices.extend(group)
        
        return [columns[i] for i in order_indices]
    
    def _calculate_hit_rate(self, str_rows, col_order):
        """Calculate hit rate for given column order."""
        n_rows = len(str_rows)
        if n_rows <= 1:
            return 0.0
        
        # Build strings with given column order
        strings = []
        for row in str_rows:
            strings.append(''.join(row[i] for i in col_order))
        
        # Calculate hit rate
        total_lcp = 0
        total_len = 0
        
        # Use a trie for efficient LCP calculation
        class TrieNode:
            def __init__(self):
                self.children = {}
                self.count = 0
        
        root = TrieNode()
        
        for i in range(n_rows):
            s = strings[i]
            total_len += len(s)
            
            if i > 0:
                # Find max LCP with previous strings
                max_lcp = 0
                node = root
                for j, ch in enumerate(s):
                    if ch in node.children:
                        node = node.children[ch]
                        max_lcp = j + 1
                    else:
                        break
                total_lcp += max_lcp
            
            # Insert current string into trie
            node = root
            for ch in s:
                if ch not in node.children:
                    node.children[ch] = TrieNode()
                node = node.children[ch]
                node.count += 1
        
        return total_lcp / total_len if total_len > 0 else 0.0
    
    def _calculate_hit_rate_sample(self, str_rows, col_order, sample_size, total_cols):
        """Calculate hit rate using sample of rows."""
        n_rows = len(str_rows)
        if n_rows <= 1:
            return 0.0
        
        # Use first sample_size rows or all rows if fewer
        sample_rows = min(sample_size, n_rows)
        
        # Build strings for sample
        strings = []
        for i in range(sample_rows):
            strings.append(''.join(str_rows[i][j] for j in col_order))
        
        # Calculate hit rate for sample
        total_lcp = 0
        total_len = 0
        
        # Simple LCP calculation for small sample
        for i in range(1, sample_rows):
            s1 = strings[i]
            total_len += len(s1)
            
            # Find max LCP with previous strings
            max_lcp = 0
            for j in range(i):
                lcp_val = 0
                min_len = min(len(s1), len(strings[j]))
                for k in range(min_len):
                    if s1[k] == strings[j][k]:
                        lcp_val += 1
                    else:
                        break
                max_lcp = max(max_lcp, lcp_val)
            
            total_lcp += max_lcp
        
        # Add length for first string
        if sample_rows > 0:
            total_len += len(strings[0])
        
        return total_lcp / total_len if total_len > 0 else 0.0
    
    def _estimate_column_gain(self, col_idx, prefix, col_stats):
        """Estimate the gain of adding a column to the prefix."""
        if not prefix:
            # First column gain based on distinct ratio
            return 1.0 - col_stats['distinct_ratios'][col_idx]
        
        # Estimate based on dependency with last column in prefix
        last_col = prefix[-1]
        return col_stats['prefix_scores'][last_col, col_idx]