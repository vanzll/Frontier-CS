import pandas as pd
import numpy as np
from collections import defaultdict, deque
from typing import List, Tuple, Dict, Set
import heapq
import math
from itertools import permutations, combinations
import random
from functools import lru_cache
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
        
        # If dataframe is empty or has 1 column, return as is
        if len(df.columns) <= 1:
            return df
        
        # Convert all values to strings
        str_df = df.astype(str)
        
        # Get column names
        columns = list(str_df.columns)
        n_cols = len(columns)
        
        # If small number of columns, try all permutations (up to 8! = 40320)
        if n_cols <= 8:
            best_order = self._exhaustive_search(str_df, columns, early_stop)
        else:
            # Use heuristic approach for larger column counts
            best_order = self._heuristic_search(
                str_df, columns, early_stop, row_stop, 
                distinct_value_threshold, parallel
            )
        
        # Reorder dataframe columns
        return df[best_order]
    
    def _apply_column_merges(self, df: pd.DataFrame, col_merge: list) -> pd.DataFrame:
        """Apply column merges by concatenating specified columns."""
        result_df = df.copy()
        
        for merge_group in col_merge:
            if not merge_group:
                continue
                
            # Ensure all columns in group exist
            valid_columns = [col for col in merge_group if col in result_df.columns]
            if len(valid_columns) <= 1:
                continue
                
            # Create merged column
            merged_name = "_".join(valid_columns)
            result_df[merged_name] = result_df[valid_columns].apply(
                lambda row: "".join(map(str, row)), axis=1
            )
            
            # Remove original columns
            result_df = result_df.drop(columns=valid_columns)
        
        return result_df
    
    def _exhaustive_search(self, str_df: pd.DataFrame, columns: List[str], 
                          early_stop: int) -> List[str]:
        """Try all permutations for small number of columns."""
        best_order = columns
        best_score = self._calculate_hit_rate(str_df, columns)
        
        # Generate permutations
        perms = list(permutations(columns))
        if len(perms) > early_stop:
            perms = random.sample(perms, min(early_stop, len(perms)))
        
        for perm in perms:
            if perm == tuple(columns):
                continue
                
            score = self._calculate_hit_rate(str_df, list(perm))
            if score > best_score:
                best_score = score
                best_order = list(perm)
        
        return best_order
    
    def _heuristic_search(self, str_df: pd.DataFrame, columns: List[str],
                         early_stop: int, row_stop: int,
                         distinct_value_threshold: float, 
                         parallel: bool) -> List[str]:
        """Heuristic search for column ordering."""
        n_rows = len(str_df)
        n_cols = len(columns)
        
        # Sample rows for faster evaluation
        if row_stop < n_rows and row_stop > 0:
            sample_indices = np.random.choice(n_rows, min(row_stop, n_rows), replace=False)
            sample_df = str_df.iloc[sample_indices].reset_index(drop=True)
        else:
            sample_df = str_df
        
        # Phase 1: Analyze column properties
        col_properties = self._analyze_columns(sample_df, columns, distinct_value_threshold)
        
        # Phase 2: Greedy construction with beam search
        best_order = self._beam_search(
            sample_df, columns, col_properties, 
            beam_width=min(50, early_stop // 10)
        )
        
        # Phase 3: Local optimization
        best_order = self._local_optimization(
            str_df, best_order, iterations=min(1000, early_stop)
        )
        
        return best_order
    
    def _analyze_columns(self, str_df: pd.DataFrame, columns: List[str],
                        distinct_value_threshold: float) -> Dict:
        """Analyze column properties to guide search."""
        n_rows = len(str_df)
        properties = {}
        
        for col in columns:
            col_values = str_df[col].values
            
            # Calculate distinct ratio
            distinct_count = len(set(col_values))
            distinct_ratio = distinct_count / n_rows
            
            # Calculate prefix potential (how often values start with same prefix)
            prefix_lengths = [min(3, len(str(v))) for v in col_values]
            prefix_counts = defaultdict(int)
            for i, val in enumerate(col_values):
                prefix = str(val)[:prefix_lengths[i]]
                prefix_counts[prefix] += 1
            
            max_prefix_freq = max(prefix_counts.values()) if prefix_counts else 0
            prefix_potential = max_prefix_freq / n_rows
            
            properties[col] = {
                'distinct_ratio': distinct_ratio,
                'prefix_potential': prefix_potential,
                'is_low_distinct': distinct_ratio < distinct_value_threshold,
                'avg_length': np.mean([len(str(v)) for v in col_values])
            }
        
        return properties
    
    def _beam_search(self, str_df: pd.DataFrame, columns: List[str],
                    col_properties: Dict, beam_width: int) -> List[str]:
        """Beam search for column ordering."""
        # Start with empty order
        beam = [([], 0.0)]
        remaining_cols = set(columns)
        
        while remaining_cols and beam:
            new_beam = []
            
            for order, score in beam:
                for col in list(remaining_cols):
                    new_order = order + [col]
                    
                    # Calculate partial score
                    partial_score = self._estimate_partial_score(
                        str_df, new_order, col_properties
                    )
                    
                    new_beam.append((new_order, partial_score))
            
            # Keep top beam_width candidates
            new_beam.sort(key=lambda x: x[1], reverse=True)
            beam = new_beam[:beam_width]
            
            # Update remaining columns
            if beam:
                remaining_cols = set(columns) - set(beam[0][0])
        
        return beam[0][0] if beam else columns
    
    def _estimate_partial_score(self, str_df: pd.DataFrame, 
                               order: List[str], 
                               col_properties: Dict) -> float:
        """Estimate score for partial order."""
        if len(order) <= 1:
            return 0.0
        
        # Use column properties to estimate
        score = 0.0
        n = len(order)
        
        for i, col in enumerate(order):
            props = col_properties[col]
            # Weight earlier columns more heavily
            weight = (n - i) / n
            
            if props['is_low_distinct']:
                # Low distinct columns are good early
                score += weight * (1 - props['distinct_ratio'])
            
            if props['prefix_potential'] > 0.5:
                # High prefix potential is good
                score += weight * props['prefix_potential']
        
        return score / len(order)
    
    def _local_optimization(self, str_df: pd.DataFrame, 
                           initial_order: List[str], 
                           iterations: int) -> List[str]:
        """Local optimization through swaps and rotations."""
        current_order = initial_order.copy()
        current_score = self._calculate_hit_rate(str_df, current_order)
        n_cols = len(current_order)
        
        for _ in range(iterations):
            improved = False
            
            # Try random swaps
            for _ in range(min(100, n_cols * 2)):
                i, j = random.sample(range(n_cols), 2)
                new_order = current_order.copy()
                new_order[i], new_order[j] = new_order[j], new_order[i]
                
                new_score = self._calculate_hit_rate(str_df, new_order)
                
                if new_score > current_score:
                    current_order = new_order
                    current_score = new_score
                    improved = True
                    break
            
            if not improved:
                # Try moving a column to different position
                for _ in range(min(50, n_cols)):
                    i = random.randint(0, n_cols - 1)
                    j = random.randint(0, n_cols - 1)
                    if i == j:
                        continue
                    
                    new_order = current_order.copy()
                    col = new_order.pop(i)
                    new_order.insert(j, col)
                    
                    new_score = self._calculate_hit_rate(str_df, new_order)
                    
                    if new_score > current_score:
                        current_order = new_order
                        current_score = new_score
                        improved = True
                        break
            
            if not improved:
                break
        
        return current_order
    
    def _calculate_hit_rate(self, str_df: pd.DataFrame, 
                           column_order: List[str]) -> float:
        """Calculate hit rate for given column order."""
        n_rows = len(str_df)
        if n_rows <= 1:
            return 0.0
        
        # Create concatenated strings
        strings = []
        total_length = 0
        
        for _, row in str_df[column_order].iterrows():
            s = "".join(row.values)
            strings.append(s)
            total_length += len(s)
        
        if total_length == 0:
            return 0.0
        
        # Calculate prefix hit rate
        total_lcp = 0
        prefix_trie = {}
        
        for i in range(n_rows):
            if i == 0:
                # Build trie for first string
                node = prefix_trie
                for char in strings[0]:
                    if char not in node:
                        node[char] = {}
                    node = node[char]
                continue
            
            # Find LCP with previous strings using trie
            s = strings[i]
            lcp = 0
            node = prefix_trie
            
            for char in s:
                if char in node:
                    lcp += 1
                    node = node[char]
                else:
                    # Add remaining suffix to trie
                    for remaining_char in s[lcp:]:
                        node[remaining_char] = {}
                        node = node[remaining_char]
                    break
            
            total_lcp += lcp
        
        return total_lcp / total_length if total_length > 0 else 0.0