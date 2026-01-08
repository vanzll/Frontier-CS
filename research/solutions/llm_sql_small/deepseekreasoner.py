import pandas as pd
import random

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
        # Step 1: Apply column merges if provided
        if col_merge is not None:
            df = self._apply_column_merges(df, col_merge)

        # If only one column, no reordering needed
        if df.shape[1] <= 1:
            return df

        # Step 2: Prepare evaluation dataset (sample if row_stop > 0)
        if 0 < row_stop < len(df):
            df_eval = df.head(row_stop)
        else:
            df_eval = df

        # Precompute string representations for all columns in the evaluation set
        col_strings = {col: df_eval[col].astype(str).values for col in df.columns}

        # Step 3: Compute initial column order based on distinct value ratio
        distinct_ratio = {}
        for col in df.columns:
            distinct_ratio[col] = df[col].nunique() / len(df)
        # Sort columns by distinct ratio ascending, then by original position
        col_pos = {col: i for i, col in enumerate(df.columns)}
        initial_order = sorted(df.columns, key=lambda c: (distinct_ratio[c], col_pos[c]))

        # Step 4: Hill climbing local search
        best_order = initial_order
        best_score = self._compute_total_lcp(col_strings, best_order)

        no_improve = 0
        iterations = 0
        max_iter = 5000  # safety limit

        while no_improve < early_stop and iterations < max_iter:
            # Generate neighbor by swapping two random columns
            neighbor = best_order.copy()
            i, j = random.sample(range(len(neighbor)), 2)
            neighbor[i], neighbor[j] = neighbor[j], neighbor[i]

            score = self._compute_total_lcp(col_strings, neighbor)
            if score > best_score:
                best_score = score
                best_order = neighbor
                no_improve = 0
            else:
                no_improve += 1
            iterations += 1

        # Return DataFrame with optimized column order
        return df[best_order]

    def _apply_column_merges(self, df: pd.DataFrame, col_merge: list) -> pd.DataFrame:
        """
        Merge groups of columns into single columns by concatenating their string values.
        """
        df_work = df.copy()
        for group in col_merge:
            # Ensure all columns in the group exist
            valid_cols = [c for c in group if c in df_work.columns]
            if not valid_cols:
                continue
            # Merge columns: concatenate values as strings
            merged_vals = df_work[valid_cols[0]].astype(str)
            for col in valid_cols[1:]:
                merged_vals += df_work[col].astype(str)
            # Create a new column name (unique)
            merged_name = "&".join(valid_cols)
            # If name already exists, append a number (should not happen with disjoint groups)
            while merged_name in df_work.columns:
                merged_name += "_"
            df_work[merged_name] = merged_vals
            # Drop the original columns
            df_work = df_work.drop(columns=valid_cols)
        return df_work

    def _compute_total_lcp(self, col_strings: dict, order: list) -> int:
        """
        Compute sum of max LCPs for the given column order using a trie.
        col_strings: dict mapping column name to array of string values for all rows.
        order: list of column names in the desired order.
        """
        total = 0
        root = {}
        num_rows = len(next(iter(col_strings.values())))  # number of rows

        for i in range(num_rows):
            # Build the concatenated string for this row
            s_parts = [col_strings[col][i] for col in order]
            s = ''.join(s_parts)

            node = root
            lcp_len = 0
            # Traverse existing trie as far as possible
            for ch in s:
                if ch in node:
                    node = node[ch]
                    lcp_len += 1
                else:
                    # Insert the missing character and the rest of the string
                    node[ch] = {}
                    new_node = node[ch]
                    for rest_ch in s[lcp_len+1:]:
                        new_node[rest_ch] = {}
                        new_node = new_node[rest_ch]
                    break
            # If the loop completed without break, the whole string is already in the trie
            total += lcp_len
        return total