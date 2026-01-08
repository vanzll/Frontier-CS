import pandas as pd
import os
from multiprocessing import Pool, cpu_count
from itertools import chain

# --- Multiprocessing worker setup ---
# Global variable to hold the DataFrame for worker processes.
# This avoids pickling the DataFrame for every task, which is slow and memory-intensive.
worker_df = None

def init_worker(df_to_share: pd.DataFrame):
    """Initializes the worker process with a global DataFrame."""
    global worker_df
    worker_df = df_to_share

def get_score_worker(perm: tuple) -> int:
    """
    Calculates the heuristic score for a given column permutation.
    The score is the negative number of unique rows/prefixes, aiming to find
    permutations that group rows together effectively.
    """
    if not perm:
        return 0
    # The list() conversion is necessary because perm is a tuple for hashing.
    return -worker_df[list(perm)].drop_duplicates().shape[0]

def calculate_lcp_sum_worker(perm: list) -> int:
    """
    Calculates the total Longest Common Prefix (LCP) sum for a given permutation.
    This is the true objective function we want to maximize.
    """
    if not perm:
        return 0
    
    # Using .values.sum(axis=1) on string arrays is a fast way to concatenate.
    try:
        # This is generally the fastest method for string concatenation in pandas/numpy
        concatenated_strings = worker_df[perm].values.sum(axis=1)
    except TypeError: # Fallback for older pandas/numpy where .sum doesn't work on strings
        arr = worker_df[perm].to_numpy()
        concatenated_strings = ["".join(row) for row in arr]

    # Sorting the strings allows calculating the total LCP sum by comparing
    # only adjacent elements, which is much faster than an N^2 comparison.
    concatenated_strings.sort()
    
    total_lcp = 0
    # os.path.commonprefix is implemented in C and can be faster than a pure Python loop.
    for i in range(1, len(concatenated_strings)):
        p = os.path.commonprefix([concatenated_strings[i-1], concatenated_strings[i]])
        total_lcp += len(p)
        
    return total_lcp

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
        Reorders columns in the DataFrame to maximize prefix hit rate.
        The method uses a beam search algorithm to find an optimal column order.
        """
        work_df, candidate_cols = self._preprocess(df, col_merge)

        if not candidate_cols:
            return work_df

        # Use a subset of the data for the search phase to speed up calculations
        df_for_search = work_df.head(early_stop)

        # Use a beam search to find a set of good candidate permutations
        candidate_perms = self._find_best_perms_beam_search(
            df_for_search, candidate_cols, col_stop, parallel
        )

        # From the candidates, select the one with the best true LCP score on the full data
        best_perm = self._select_best_permutation(
            work_df, candidate_perms, parallel
        )

        return work_df[best_perm]

    def _preprocess(self, df: pd.DataFrame, col_merge: list) -> (pd.DataFrame, list):
        """
        Handles column merges and prepares the DataFrame for processing.
        Returns a new DataFrame with merged columns and a list of columns to reorder.
        All columns are converted to string type.
        """
        if not col_merge:
            work_df = df.astype(str)
            return work_df, df.columns.tolist()

        df_merged = df.copy()
        
        merged_source_cols = set(chain.from_iterable(col_merge))
        unmerged_cols = [c for c in df.columns if c not in merged_source_cols]
        
        merged_cols = []
        for i, group in enumerate(col_merge):
            if not group: continue
            # Create a unique name for the new merged column
            new_col_name = f"__merged_{'_'.join(map(str, group))}"
            k = 0
            while new_col_name in df_merged.columns:
                k += 1
                new_col_name = f"__merged_{'_'.join(map(str, group))}_{k}"
            
            df_merged[new_col_name] = df_merged[group].astype(str).agg("".join, axis=1)
            merged_cols.append(new_col_name)

        candidate_cols = unmerged_cols + merged_cols
        # The working dataframe contains only the columns to be ordered
        work_df = df_merged[candidate_cols].astype(str)

        return work_df, candidate_cols

    def _find_best_perms_beam_search(self, df: pd.DataFrame, candidate_cols: list, beam_width: int, parallel: bool) -> list:
        """
        Performs a beam search to find the best column permutations.
        The search is guided by a heuristic score (number of unique prefixes).
        """
        M = len(candidate_cols)
        if M == 0:
            return [[]]
        
        # Beam stores tuples of (score, permutation)
        beam = [(0, [])]

        for _ in range(M):
            next_candidates = []
            for _, perm in beam:
                remaining_cols = [c for c in candidate_cols if c not in perm]
                for c in remaining_cols:
                    # Permutations are tuples to be hashable for use in sets
                    next_candidates.append(tuple(perm + [c]))
            
            if not next_candidates: break
            
            # Remove duplicates that might arise from different search paths
            next_candidates = list(set(next_candidates))
            
            scores = []
            if parallel and len(next_candidates) > 1:
                try:
                    n_procs = min(cpu_count(), len(next_candidates))
                    with Pool(processes=n_procs, initializer=init_worker, initargs=(df,)) as pool:
                        scores = pool.map(get_score_worker, next_candidates, chunksize=max(1, len(next_candidates) // n_procs))
                except Exception: # Fallback to sequential on any multiprocessing issue
                    parallel = False

            if not parallel:
                global worker_df
                worker_df = df
                scores = [get_score_worker(p) for p in next_candidates]
            
            scored_candidates = sorted(zip(scores, next_candidates), key=lambda x: x[0], reverse=True)
            beam = scored_candidates[:beam_width]

        return [list(p) for _, p in beam]

    def _select_best_permutation(self, df: pd.DataFrame, perms: list, parallel: bool) -> list:
        """
        Selects the best permutation from a list of candidates by calculating the true LCP sum.
        """
        if not perms:
            return []
        if len(perms) == 1:
            return perms[0]

        lcp_sums = []
        if parallel and len(perms) > 1:
            try:
                n_procs = min(cpu_count(), len(perms))
                with Pool(processes=n_procs, initializer=init_worker, initargs=(df,)) as pool:
                    lcp_sums = pool.map(calculate_lcp_sum_worker, perms, chunksize=1)
            except Exception: # Fallback to sequential
                parallel = False
        
        if not parallel:
            global worker_df
            worker_df = df
            lcp_sums = [calculate_lcp_sum_worker(p) for p in perms]

        if not lcp_sums:
            return perms[0]
            
        best_idx = lcp_sums.index(max(lcp_sums))
        return perms[best_idx]