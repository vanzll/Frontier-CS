import pandas as pd
from joblib import Parallel, delayed

def _get_score(df: pd.DataFrame, perm: list) -> int:
    """
    Calculates the number of unique rows for a given subset of columns.
    This serves as the score for a permutation, where lower is better,
    as it indicates less diversity and more potential for prefix matching.
    """
    if not perm:
        return 1
    return df[perm].drop_duplicates().shape[0]

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
        if df.empty:
            return df

        # Create a working copy of the DataFrame to modify
        df_reordered = df.copy()

        # 1. Handle column merges if specified
        if col_merge:
            for i, group in enumerate(col_merge):
                # Ensure all columns in the group exist before merging
                if not all(c in df_reordered.columns for c in group):
                    continue
                
                new_col_name = f'merged_{"_".join(map(str, group))}'
                df_reordered[new_col_name] = df_reordered[group].astype(str).agg(''.join, axis=1)
                df_reordered = df_reordered.drop(columns=group)
        
        cols = df_reordered.columns.tolist()
        if len(cols) <= 1:
            return df_reordered

        # 2. Convert all columns to string for consistent processing
        df_processed = df_reordered.astype(str)

        # 3. Create a sample for faster heuristic calculations
        sample_size = 5000
        if len(df_processed) > sample_size:
            df_sample = df_processed.sample(n=sample_size, random_state=42)
        else:
            df_sample = df_processed
        
        # 4. Identify high-cardinality ("bad") columns
        num_rows_sample = len(df_sample)
        good_cols = []
        bad_cols = []
        
        if num_rows_sample > 0:
            distinct_counts = df_sample.nunique()
            for c in cols:
                if distinct_counts[c] / num_rows_sample > distinct_value_threshold:
                    bad_cols.append(c)
                else:
                    good_cols.append(c)
            # Sort bad columns by their distinct count, less bad first
            bad_cols.sort(key=lambda c: distinct_counts[c])
        else:
            # If sample is empty, treat all columns as good
            good_cols = cols

        best_good_perm = []
        if good_cols:
            n_jobs = 8 if parallel else 1
            
            # 5. Beam search for the best prefix of `good_cols`
            
            # Step 1: Initialize beams with single columns
            if num_rows_sample > 0:
                initial_candidates = [([c], distinct_counts[c]) for c in good_cols]
            else:
                initial_candidates = [([c], 1) for c in good_cols]

            initial_candidates.sort(key=lambda x: x[1])
            beams = [item for item in initial_candidates if item[1] < early_stop][:col_stop]

            # Steps 2 to `row_stop`: Extend beams
            search_depth = min(row_stop, len(good_cols))
            for _ in range(1, search_depth):
                if not beams:
                    break
                
                tasks = []
                for p, _ in beams:
                    remaining = [c for c in good_cols if c not in p]
                    for c in remaining:
                        tasks.append(p + [c])

                if not tasks:
                    break
                
                scores = Parallel(n_jobs=n_jobs)(delayed(_get_score)(df_sample, p) for p in tasks)
                
                candidates = []
                for p, score in zip(tasks, scores):
                    if score < early_stop:
                        candidates.append((p, score))

                if not candidates:
                    beams = []
                    break
                
                candidates.sort(key=lambda x: x[1])
                beams = candidates[:col_stop]
            
            best_prefix = beams[0][0] if beams else []

            # 6. Greedy completion for the rest of `good_cols`
            current_perm = best_prefix
            remaining_cols = [c for c in good_cols if c not in current_perm]
            
            while remaining_cols:
                if len(remaining_cols) == 1:
                    current_perm.append(remaining_cols[0])
                    break
                
                tasks = [current_perm + [c] for c in remaining_cols]
                scores = Parallel(n_jobs=n_jobs)(delayed(_get_score)(df_sample, p) for p in tasks)
                
                best_idx = min(range(len(scores)), key=scores.__getitem__)
                best_next_col = remaining_cols.pop(best_idx)
                current_perm.append(best_next_col)

            best_good_perm = current_perm

        # 7. Final permutation is the ordered good columns followed by ordered bad columns
        final_perm = best_good_perm + bad_cols
        
        return df_reordered[final_perm]