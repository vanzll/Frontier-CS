import pandas as pd
import numpy as np
from collections import Counter
from multiprocessing import Pool, cpu_count
from typing import List, Tuple
import itertools

def compute_sum_lcp_for_permutation(perm: Tuple[int, ...], data: List[List[str]]) -> int:
    """
    Compute the sum of LCPs for a given permutation of column indices.
    data: list of rows, each row is a list of strings for each column.
    """
    trie = {}
    total_lcp = 0
    for row in data:
        # Build the concatenated string for this row in the given column order
        s = ''.join(row[i] for i in perm)
        # Traverse trie to compute LCP
        node = trie
        lcp = 0
        for ch in s:
            if ch in node:
                lcp += 1
                node = node[ch]
            else:
                break
        total_lcp += lcp
        # Insert the whole string into the trie
        node = trie
        for ch in s:
            if ch not in node:
                node[ch] = {}
            node = node[ch]
    return total_lcp

def generate_neighbors(current_order: List[int], col_stop: int) -> List[Tuple[int, ...]]:
    """Generate neighbor permutations by swapping two columns.
    If col_stop < len(current_order), only swap columns in first col_stop positions with others.
    """
    k = len(current_order)
    neighbors = []
    if col_stop >= k:
        # Consider all swaps
        for i in range(k):
            for j in range(i + 1, k):
                new_order = current_order.copy()
                new_order[i], new_order[j] = new_order[j], new_order[i]
                neighbors.append(tuple(new_order))
    else:
        # Only swap columns in the first col_stop positions with any later column
        for i in range(col_stop):
            for j in range(i + 1, k):
                new_order = current_order.copy()
                new_order[i], new_order[j] = new_order[j], new_order[i]
                neighbors.append(tuple(new_order))
    return neighbors

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
        # Step 1: Apply column merges
        df = self._apply_column_merges(df, col_merge)
        
        # If only one column, no reordering needed
        if len(df.columns) <= 1:
            return df
        
        # Step 2: Prepare sample rows for evaluation
        sample_rows = min(row_stop, len(df))
        sample_indices = list(range(sample_rows))
        columns = list(df.columns)
        k = len(columns)
        
        # Precompute string data for sample rows
        sample_data = []
        for idx in sample_indices:
            row = df.iloc[idx]
            sample_data.append([str(row[col]) for col in columns])
        
        # Step 3: Compute initial order using heuristic
        initial_order = self._compute_initial_order(df, columns, sample_indices)
        
        # Step 4: Hill climbing with swaps
        best_order = self._hill_climb(
            initial_order, sample_data, early_stop, col_stop, parallel
        )
        
        # Step 5: Return DataFrame with best column order
        best_columns = [columns[i] for i in best_order]
        return df[best_columns]
    
    def _apply_column_merges(self, df: pd.DataFrame, col_merge: list) -> pd.DataFrame:
        """Merge columns according to col_merge specification."""
        if col_merge is None:
            return df.copy()
        df = df.copy()
        for group in col_merge:
            if not group:
                continue
            # Use first column name as the merged column name
            new_col = group[0]
            # Concatenate values in the group
            def merge_func(row):
                return ''.join(str(row[col]) for col in group)
            df[new_col] = df.apply(merge_func, axis=1)
            # Drop original columns in the group (except the one we kept)
            to_drop = [col for col in group if col != new_col and col in df.columns]
            df.drop(columns=to_drop, inplace=True)
        return df
    
    def _compute_initial_order(self, df: pd.DataFrame, columns: List[str], sample_indices: List[int]) -> List[int]:
        """Compute initial column order based on heuristic score."""
        k = len(columns)
        scores = []
        for col_idx, col in enumerate(columns):
            # Use sample rows to compute statistics
            values = [str(df.iloc[i][col]) for i in sample_indices]
            avg_len = np.mean([len(v) for v in values])
            # Frequency of the most common value
            if values:
                most_common_count = max(Counter(values).values()) / len(values)
            else:
                most_common_count = 0.0
            score = avg_len * most_common_count
            scores.append((col_idx, score))
        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)
        return [idx for idx, _ in scores]
    
    def _hill_climb(
        self,
        initial_order: List[int],
        sample_data: List[List[str]],
        early_stop: int,
        col_stop: int,
        parallel: bool,
    ) -> List[int]:
        """Hill climbing local search to improve column order."""
        current_order = initial_order.copy()
        current_score = compute_sum_lcp_for_permutation(tuple(current_order), sample_data)
        no_improve = 0
        k = len(current_order)
        
        while no_improve < early_stop:
            # Generate neighbor permutations
            neighbors = generate_neighbors(current_order, col_stop)
            if not neighbors:
                break
            
            # Evaluate neighbors
            neighbor_scores = []
            if parallel and len(neighbors) > 1:
                # Use multiprocessing
                with Pool(processes=min(cpu_count(), len(neighbors))) as pool:
                    # Prepare arguments: each neighbor tuple and the shared sample_data
                    args = [(neighbor, sample_data) for neighbor in neighbors]
                    results = pool.starmap(compute_sum_lcp_for_permutation, args)
                neighbor_scores = results
            else:
                for neighbor in neighbors:
                    score = compute_sum_lcp_for_permutation(neighbor, sample_data)
                    neighbor_scores.append(score)
            
            # Find best neighbor
            best_idx = np.argmax(neighbor_scores)
            best_neighbor_score = neighbor_scores[best_idx]
            
            if best_neighbor_score > current_score:
                current_order = list(neighbors[best_idx])
                current_score = best_neighbor_score
                no_improve = 0
            else:
                no_improve += 1
        
        return current_order