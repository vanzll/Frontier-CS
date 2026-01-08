import pandas as pd
import numpy as np
from typing import List, Any, Dict, Tuple, Optional


class Solution:
    def _unique_col_name(self, df: pd.DataFrame, base: str) -> str:
        if base not in df.columns:
            return base
        i = 2
        while f"{base}__{i}" in df.columns:
            i += 1
        return f"{base}__{i}"

    def _normalize_merge_group(self, group: Any, orig_cols: List[Any]) -> List[Any]:
        if group is None:
            return []
        if not isinstance(group, (list, tuple)):
            group = [group]
        cols = []
        for x in group:
            if isinstance(x, (int, np.integer)):
                if 0 <= int(x) < len(orig_cols):
                    cols.append(orig_cols[int(x)])
            else:
                cols.append(x)
        return cols

    def _apply_merges(self, df: pd.DataFrame, col_merge: list) -> pd.DataFrame:
        if not col_merge:
            return df
        orig_cols = list(df.columns)
        for group in col_merge:
            cols = self._normalize_merge_group(group, orig_cols)
            cols = [c for c in cols if c in df.columns]
            if len(cols) <= 1:
                continue
            loc = df.columns.get_loc(cols[0])
            base_name = "MERGE__" + "__".join(str(c) for c in cols)
            new_col = self._unique_col_name(df, base_name)

            s = df[cols[0]].astype(str)
            for c in cols[1:]:
                s = s.str.cat(df[c].astype(str), sep="")
            df.insert(loc, new_col, s.to_numpy())
            df.drop(columns=cols, inplace=True, errors="ignore")
        return df

    def _col_stats(self, s: pd.Series, n2: float) -> Dict[str, float]:
        s_str = s.astype(str)
        lens = s_str.str.len().to_numpy(dtype=np.float32, copy=False)
        avg_len = float(lens.mean()) if lens.size else 0.0

        codes_full, _ = pd.factorize(s_str, sort=False)
        if codes_full.size == 0:
            return {
                "avg_len": 0.0,
                "distinct_ratio": 1.0,
                "sim_full": 0.0,
                "sim1": 0.0,
                "sim2": 0.0,
                "sim4": 0.0,
                "top_ratio": 0.0,
                "score": 0.0,
            }

        counts_full = np.bincount(codes_full)
        distinct_ratio = float(counts_full.size) / float(codes_full.size)
        sim_full = float(np.dot(counts_full, counts_full)) / n2
        top_ratio = float(counts_full.max()) / float(codes_full.size)

        def sim_prefix(k: int) -> float:
            if k <= 0:
                return 0.0
            pref = s_str.str.slice(0, k)
            codes, _ = pd.factorize(pref, sort=False)
            counts = np.bincount(codes)
            return float(np.dot(counts, counts)) / n2

        sim1 = sim_prefix(1)
        sim2 = sim_prefix(2)
        sim4 = sim_prefix(4)

        est = sim1 + sim2 + sim4 + max(avg_len - 4.0, 0.0) * sim_full
        penalty = (distinct_ratio + 1e-9) ** 0.30
        score = est / penalty

        return {
            "avg_len": avg_len,
            "distinct_ratio": distinct_ratio,
            "sim_full": sim_full,
            "sim1": sim1,
            "sim2": sim2,
            "sim4": sim4,
            "top_ratio": top_ratio,
            "score": float(score),
        }

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
        if df is None or df.shape[1] <= 1:
            return df

        df2 = df.copy()
        df2 = self._apply_merges(df2, col_merge)

        cols = list(df2.columns)
        if len(cols) <= 1:
            return df2

        n_total = len(df2)
        if n_total == 0:
            return df2

        n_stat = min(n_total, int(early_stop) if early_stop is not None else n_total, 20000)
        n_stat = max(2000, n_stat) if n_total >= 2000 else n_total
        df_stat = df2.iloc[:n_stat]
        n2 = float(n_stat) * float(n_stat)

        stats: Dict[Any, Dict[str, float]] = {}
        for c in cols:
            stats[c] = self._col_stats(df_stat[c], n2)

        def sort_key(c: Any) -> Tuple[float, float, float, float, float]:
            st = stats[c]
            distinct_ratio = st["distinct_ratio"]
            score = st["score"]
            sim_full = st["sim_full"]
            top_ratio = st["top_ratio"]
            avg_len = st["avg_len"]

            # Push near-unique columns toward the end unless they have meaningful shared prefixes
            uniqueness_pen = 0.0
            if distinct_ratio >= 0.98 and score < 3.0:
                uniqueness_pen = 1.0

            # Slight extra bias towards lower distinct ratio in early positions
            distinct_bias = -distinct_ratio

            return (-uniqueness_pen, score, sim_full, top_ratio, avg_len + 0.001 * distinct_bias)

        ordered = sorted(cols, key=sort_key, reverse=True)
        return df2.loc[:, ordered]