import pandas as pd
from collections import defaultdict
from typing import List, Tuple, Dict, Any


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
        df = self._apply_col_merge(df, col_merge)

        # Step 2: Prepare data (convert to strings, lengths, distinctness)
        values, lengths, distinct_ratio, col_names = self._prepare_columns(df)

        # Step 3: Choose column order using greedy grouped-trie marginal LCP heuristic
        n_rows = min(len(df), max(1, early_stop))
        order = self._choose_order(values, n_rows)

        # Step 4: Return DataFrame with columns reordered
        ordered_cols = [col_names[i] for i in order]
        return df.loc[:, ordered_cols]

    def _apply_col_merge(self, df: pd.DataFrame, col_merge: list) -> pd.DataFrame:
        if not col_merge:
            return df

        df = df.copy()
        used_cols = set()
        new_cols_data = []
        new_cols_names = []

        # Helper to normalize column identifiers to names
        def normalize(col):
            if isinstance(col, int):
                return df.columns[col]
            return col

        for idx, group in enumerate(col_merge):
            if not group:
                continue
            group_names = [normalize(c) for c in group if normalize(c) in df.columns]
            if not group_names:
                continue
            for c in group_names:
                used_cols.add(c)

            # Build merged column by concatenating string representations per row
            n = len(df)
            merged_vals = [''] * n
            for cname in group_names:
                arr = df[cname].to_numpy()
                mask = pd.isna(arr)
                # append string
                for i in range(n):
                    v = '' if mask[i] else str(arr[i])
                    merged_vals[i] += v

            new_col_name = f"MERGED_{idx}"
            new_cols_names.append(new_col_name)
            new_cols_data.append(pd.Series(merged_vals, index=df.index))

        # Drop merged columns
        remaining_cols = [c for c in df.columns if c not in used_cols]
        # Append new merged columns at the end
        for name, series in zip(new_cols_names, new_cols_data):
            df[name] = series

        # Keep the order: remaining original columns followed by new merged columns
        return df.loc[:, remaining_cols + new_cols_names]

    def _prepare_columns(self, df: pd.DataFrame):
        col_names = list(df.columns)
        N = len(df)
        values = []
        lengths = []
        distinct_ratio = []

        for cname in col_names:
            arr = df[cname].to_numpy()
            mask = pd.isna(arr)
            col_vals = [''] * N
            col_lens = [0] * N
            uniq_set = set()
            for i in range(N):
                if mask[i]:
                    s = ''
                else:
                    s = str(arr[i])
                col_vals[i] = s
                col_lens[i] = len(s)
                uniq_set.add(s)
            values.append(col_vals)
            lengths.append(col_lens)
            distinct_ratio.append(len(uniq_set) / N if N > 0 else 0.0)

        return values, lengths, distinct_ratio, col_names

    def _choose_order(self, values: List[List[str]], n_rows: int) -> List[int]:
        M = len(values)
        if M <= 1:
            return list(range(M))

        # Restrict values to early_stop rows
        vals = [col_vals[:n_rows] for col_vals in values]

        # Initialize all rows into a single group id 0
        group_ids = [0] * n_rows

        remaining = set(range(M))
        order = []

        # Greedy selection of columns
        for _ in range(M):
            best_col = None
            best_score = -1

            # Evaluate marginal LCP for each candidate column under current groups
            for c in remaining:
                score = self._marginal_lcp_grouped(group_ids, vals[c])
                if score > best_score:
                    best_score = score
                    best_col = c

            if best_col is None:
                # Fallback: if something went wrong, append remaining arbitrarily
                order.extend(sorted(list(remaining)))
                break

            order.append(best_col)
            remaining.remove(best_col)

            # Update group ids by extending with selected column's values
            group_ids = self._update_groups(group_ids, vals[best_col])

        return order

    def _marginal_lcp_grouped(self, group_ids: List[int], col_vals: List[str]) -> int:
        # Build separate tries for each group; compute LCP against prior rows in the same group
        roots: Dict[int, dict] = {}
        total_lcp = 0
        # local variables for speed
        gid_list = group_ids
        vals = col_vals
        get_root = roots.get
        for i in range(len(vals)):
            g = gid_list[i]
            s = vals[i]
            root = get_root(g)
            if root is None:
                root = {}
                roots[g] = root

            # Match phase
            node = root
            matched = 0
            for ch in s:
                nxt = node.get(ch)
                if nxt is None:
                    break
                matched += 1
                node = nxt
            total_lcp += matched

            # Insert phase
            node = root
            for ch in s:
                nxt = node.get(ch)
                if nxt is None:
                    nxt = {}
                    node[ch] = nxt
                node = nxt

        return total_lcp

    def _update_groups(self, group_ids: List[int], col_vals: List[str]) -> List[int]:
        # Map (prev_group, value) -> new_group_id
        pair_to_id: Dict[Tuple[int, str], int] = {}
        next_id = 0
        new_group_ids = [0] * len(group_ids)
        for i in range(len(group_ids)):
            key = (group_ids[i], col_vals[i])
            g = pair_to_id.get(key)
            if g is None:
                g = next_id
                pair_to_id[key] = g
                next_id += 1
            new_group_ids[i] = g
        return new_group_ids