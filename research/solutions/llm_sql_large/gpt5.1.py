import pandas as pd
import numpy as np


class Solution:
    def _normalize_merge_group(self, group, original_columns):
        if not group:
            return []

        cols = list(original_columns)
        names = []

        # If all entries are ints, treat them as indices into the original columns
        if all(isinstance(g, int) for g in group):
            if not cols:
                return []

            max_idx = max(group)
            min_idx = min(group)

            # Decide if indices are 0-based or 1-based
            if 0 <= min_idx and max_idx < len(cols):
                # 0-based
                for idx in group:
                    if 0 <= idx < len(cols):
                        names.append(cols[idx])
            elif 1 <= min_idx and max_idx <= len(cols):
                # 1-based
                for idx in group:
                    adj = idx - 1
                    if 0 <= adj < len(cols):
                        names.append(cols[adj])
            else:
                # Indices do not match column range; ignore this group
                names = []
        else:
            # Treat entries as column names
            for g in group:
                if isinstance(g, str) and g in cols:
                    names.append(g)

        # Deduplicate while preserving order
        seen = set()
        ordered_names = []
        for c in names:
            if c not in seen:
                seen.add(c)
                ordered_names.append(c)

        return ordered_names

    def _apply_column_merges(self, df, col_merge):
        if not col_merge:
            return df

        original_columns = list(df.columns)

        for group in col_merge:
            if not group:
                continue

            merge_cols = self._normalize_merge_group(group, original_columns)
            if len(merge_cols) <= 1:
                continue

            # Only keep columns that still exist in the current DataFrame
            merge_cols = [c for c in merge_cols if c in df.columns]
            if len(merge_cols) <= 1:
                continue

            # New column name: based on merged columns, ensure uniqueness
            base_name = "||".join(str(c) for c in merge_cols)
            new_name = base_name
            suffix = 1
            existing = set(df.columns)
            while new_name in existing:
                new_name = f"{base_name}_m{suffix}"
                suffix += 1

            # Concatenate string representations of the merge columns
            df[new_name] = df[merge_cols].astype(str).agg("".join, axis=1)

            # Drop original merged columns
            df.drop(columns=merge_cols, inplace=True)

        return df

    def _compute_column_scores(self, df, distinct_value_threshold):
        N = len(df)
        scores = {}

        if N == 0:
            for col in df.columns:
                scores[col] = 0.0
            return scores

        for col in df.columns:
            s = df[col]
            vals = s.values

            # Adjacency equality probability
            if N > 1:
                try:
                    eq_adj = float(np.mean(vals[1:] == vals[:-1]))
                except Exception:
                    # Fallback using pandas if direct numpy comparison fails
                    eq_adj = float((s.shift(-1) == s).iloc[:-1].mean())
            else:
                eq_adj = 1.0

            # String representations for value_counts and length
            try:
                s_str = s.astype(str)
            except Exception:
                s_str = s.map(str)

            # Average string length
            try:
                len_mean = float(s_str.str.len().mean())
            except Exception:
                lengths = [len(v) for v in s_str]
                len_mean = float(sum(lengths)) / float(N) if N > 0 else 0.0

            # Probability that two random rows have equal value in this column
            counts = s_str.value_counts(dropna=False)
            freq = counts.values.astype("float64") / float(N)
            p_allpairs = float(np.sum(freq * freq))

            # Distinct ratio based on original values
            try:
                nunique = float(s.nunique(dropna=False))
            except Exception:
                nunique = float(s_str.nunique(dropna=False))
            distinct_ratio = nunique / float(N) if N > 0 else 0.0

            # Effective equality probability, mixing global and adjacency info
            p_eff = 0.7 * p_allpairs + 0.3 * eq_adj

            # Penalize very high distinct-ratio columns
            if (
                distinct_value_threshold is not None
                and distinct_value_threshold < 1.0
                and distinct_value_threshold > 0.0
                and distinct_ratio > distinct_value_threshold
            ):
                over = min(1.0, distinct_ratio) - distinct_value_threshold
                denom = 1.0 - distinct_value_threshold + 1e-9
                factor = max(0.0, 1.0 - over / denom)
                p_eff *= factor

            # Clamp to [0, 1]
            if p_eff < 0.0:
                p_eff = 0.0
            elif p_eff > 1.0:
                p_eff = 1.0

            if len_mean <= 0.0 or p_eff <= 0.0:
                score = 0.0
            else:
                if p_eff >= 0.999999:
                    # Almost constant: very beneficial early; scale with length
                    score = 1e12 * len_mean
                else:
                    score = float((p_eff * len_mean) / (1.0 - p_eff + 1e-9))

            scores[col] = score

        return scores

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
        # Work on a copy to avoid mutating the input DataFrame
        df_work = df.copy()

        # Apply column merges before reordering
        df_work = self._apply_column_merges(df_work, col_merge)

        # Use a sample for scoring if dataset is very large
        N_total = len(df_work)
        if early_stop is not None and isinstance(early_stop, int) and early_stop > 0 and N_total > early_stop:
            # Preserve order; use first `early_stop` rows
            df_sample = df_work.iloc[:early_stop]
        else:
            df_sample = df_work

        # Compute heuristic scores for columns based on the sample
        col_scores = self._compute_column_scores(df_sample, distinct_value_threshold)

        # Determine column order by descending score
        ordered_cols = sorted(
            df_work.columns,
            key=lambda c: col_scores.get(c, 0.0),
            reverse=True,
        )

        # Return DataFrame with reordered columns
        return df_work[ordered_cols]