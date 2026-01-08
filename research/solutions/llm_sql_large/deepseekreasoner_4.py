import pandas as pd
import numpy as np
from collections import defaultdict, deque
import random
import itertools
from functools import lru_cache
import math
import time
from typing import List, Tuple, Dict, Set

class TrieNode:
    __slots__ = ('children',)
    def __init__(self):
        self.children = {}

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
        if col_merge is not None:
            df = self._apply_column_merges(df, col_merge)
        
        # Convert all values to strings
        df = df.astype(str)
        
        # Get sample for optimization
        sample_size = min(1000, len(df))
        if len(df) <= 2000:
            sample_indices = list(range(len(df)))
        else:
            random.seed(42)
            sample_indices = random.sample(range(len(df)), sample_size)
        
        # Precompute column strings for sample
        col_strings = {}
        col_lengths = {}
        for col in df.columns:
            strings = [df.iloc[i][col] for i in sample_indices]
            col_strings[col] = strings
            col_lengths[col] = sum(len(s) for s in strings)
        
        # Greedy column ordering
        remaining_cols = list(df.columns)
        ordered_cols = []
        
        # Initialize trie state
        trie_root = TrieNode()
        current_leaves = [trie_root] * len(sample_indices)
        current_lcp = 0
        current_total_len = 0
        
        # Group rows by current leaf node
        groups = defaultdict(list)
        for i, node in enumerate(current_leaves):
            groups[node].append(i)
        
        while remaining_cols:
            # Select candidate columns based on distinct value ratio
            candidates = self._select_candidates(
                remaining_cols, col_strings, sample_indices,
                distinct_value_threshold, early_stop
            )
            
            best_col = None
            best_score = -1
            best_additional_lcp = 0
            best_new_total_len = 0
            
            for col in candidates:
                additional_lcp = 0
                col_total_len = col_lengths[col]
                
                # Compute additional LCP for this column
                for leaf_node, row_indices in groups.items():
                    if len(row_indices) == 1:
                        continue
                    
                    # Build mini-trie for this group
                    mini_trie = {}
                    for idx in row_indices:
                        s = col_strings[col][idx]
                        node = mini_trie
                        depth = 0
                        for ch in s:
                            if ch in node:
                                node = node[ch]
                                depth += 1
                            else:
                                node[ch] = {}
                                node = node[ch]
                        additional_lcp += depth
                
                # Calculate score (hit rate)
                new_total_len = current_total_len + col_total_len
                if new_total_len > 0:
                    score = (current_lcp + additional_lcp) / new_total_len
                else:
                    score = 0
                
                if score > best_score:
                    best_score = score
                    best_col = col
                    best_additional_lcp = additional_lcp
                    best_new_total_len = col_total_len
            
            if best_col is None:
                break
            
            # Update state with chosen column
            ordered_cols.append(best_col)
            remaining_cols.remove(best_col)
            
            # Update trie with chosen column
            new_groups = defaultdict(list)
            for leaf_node, row_indices in groups.items():
                # For each row in this group, extend the trie
                for idx in row_indices:
                    s = col_strings[best_col][idx]
                    node = leaf_node
                    for ch in s:
                        if ch not in node.children:
                            node.children[ch] = TrieNode()
                        node = node.children[ch]
                    current_leaves[idx] = node
                    new_groups[node].append(idx)
            
            groups = new_groups
            current_lcp += best_additional_lcp
            current_total_len += best_new_total_len
        
        # Add any remaining columns in original order
        ordered_cols.extend(remaining_cols)
        
        # Return DataFrame with reordered columns
        return df[ordered_cols]
    
    def _apply_column_merges(self, df: pd.DataFrame, col_merge: list) -> pd.DataFrame:
        """Merge specified columns into single columns."""
        new_df = df.copy()
        
        for merge_group in col_merge:
            if not merge_group:
                continue
            
            # Create merged column name
            merged_name = '_'.join(str(col) for col in merge_group)
            
            # Merge columns by concatenating string values
            merged_values = []
            for i in range(len(df)):
                merged_val = ''.join(str(df.iloc[i][col]) for col in merge_group)
                merged_values.append(merged_val)
            
            new_df[merged_name] = merged_values
            
            # Remove original columns
            for col in merge_group:
                if col in new_df.columns:
                    new_df = new_df.drop(columns=[col])
        
        return new_df
    
    def _select_candidates(
        self,
        remaining_cols: List[str],
        col_strings: Dict[str, List[str]],
        sample_indices: List[int],
        distinct_value_threshold: float,
        early_stop: int
    ) -> List[str]:
        """Select candidate columns based on distinct value ratio."""
        candidates = []
        for col in remaining_cols:
            strings = col_strings[col]
            distinct_count = len(set(strings))
            distinct_ratio = distinct_count / len(sample_indices)
            
            if distinct_ratio <= distinct_value_threshold:
                candidates.append((distinct_ratio, col))
        
        # Sort by distinct ratio (lower is better)
        candidates.sort(key=lambda x: x[0])
        selected = [col for _, col in candidates[:early_stop]]
        
        # If no columns meet threshold, use all columns
        if not selected:
            selected = remaining_cols[:early_stop]
        
        return selected