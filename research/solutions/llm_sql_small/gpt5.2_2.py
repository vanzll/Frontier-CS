import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Optional


class Solution:
    _MASK64 = (1 << 64) - 1
    _P_TOK = 11400714819323198485  # 64-bit odd constant
    _P_CH = 1099511628211          # FNV-ish prime

    def _apply_merges(self, df: pd.DataFrame, col_merge: Optional[list]) -> pd.DataFrame:
        if not col_merge:
            return df.copy()

        out = df.copy()
        for group in col_merge:
            if not group:
                continue

            current_cols = list(out.columns)
            names = []
            for x in group:
                if isinstance(x, (int, np.integer)):
                    if 0 <= int(x) < len(current_cols):
                        names.append(current_cols[int(x)])
                else:
                    if x in out.columns:
                        names.append(x)

            names = [c for c in names if c in out.columns]
            if len(names) <= 1:
                continue

            insert_idx = min(list(out.columns).index(c) for c in names)

            merged = out[names[0]].astype(str)
            for c in names[1:]:
                merged = merged + out[c].astype(str)

            base_name = "__".join(str(c) for c in names)
            new_name = base_name
            k = 1
            while new_name in out.columns and new_name not in names:
                k += 1
                new_name = f"{base_name}__{k}"

            cols = list(out.columns)
            for c in names:
                cols.remove(c)
            cols.insert(insert_idx, new_name)

            out[new_name] = merged
            out = out[cols]

        return out

    def _prepare_sample_token_data(
        self, df: pd.DataFrame, cols: List, k_rows: int
    ) -> Tuple[List[List[str]], List[List[int]], List[List[int]]]:
        sample = df.iloc[:k_rows][cols].astype(str)
        col_strs = []
        col_lens = []
        col_toks = []
        for c in cols:
            s_list = sample[c].tolist()
            col_strs.append(s_list)
            col_lens.append([len(x) for x in s_list])
            col_toks.append([(hash(x) & self._MASK64) for x in s_list])
        return col_strs, col_lens, col_toks

    def _prepare_sample_char_data(
        self, df: pd.DataFrame, cols: List, k_rows: int
    ) -> Tuple[bool, List[List[bytes]], List[List[str]]]:
        sample = df.iloc[:k_rows][cols].astype(str)
        col_strs = []
        all_ascii = True
        for c in cols:
            s_list = sample[c].tolist()
            col_strs.append(s_list)
            if all_ascii:
                for s in s_list:
                    if not s.isascii():
                        all_ascii = False
                        break
        if all_ascii:
            col_bytes = [[s.encode("ascii") for s in s_list] for s_list in col_strs]
            return True, col_bytes, col_strs
        else:
            return False, [], col_strs

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
        dfm = self._apply_merges(df, col_merge)
        cols = list(dfm.columns)
        m = len(cols)
        n = len(dfm)

        if m <= 1:
            return dfm.astype(str)

        if early_stop is None:
            early_stop = n
        try:
            early_stop = int(early_stop)
        except Exception:
            early_stop = n

        k_token = min(n, max(800, min(early_stop, 5000)))
        col_strs_tok, col_lens_tok, col_toks_tok = self._prepare_sample_token_data(dfm, cols, k_token)

        def token_score(order: Tuple[int, ...], cache: Dict[Tuple[int, ...], int]) -> int:
            if order in cache:
                return cache[order]
            tok_cols = [col_toks_tok[i] for i in order]
            len_cols = [col_lens_tok[i] for i in order]
            seen = set()
            total = 0
            P = self._P_TOK
            mask = self._MASK64

            for r in range(k_token):
                h = 1469598103934665603
                miss = False
                hit_len = 0
                for tcol, lcol in zip(tok_cols, len_cols):
                    h = (h * P + tcol[r] + 1) & mask
                    if (not miss) and (h in seen):
                        hit_len += lcol[r]
                    else:
                        miss = True
                    seen.add(h)
                if r:
                    total += hit_len

            cache[order] = total
            return total

        distinct_ratios = []
        avg_lens = []
        for j in range(m):
            vals = col_strs_tok[j]
            distinct_ratios.append(len(set(vals)) / float(len(vals) if len(vals) else 1))
            avg_lens.append(sum(col_lens_tok[j]) / float(len(col_lens_tok[j]) if len(col_lens_tok[j]) else 1))

        heur_order = tuple(sorted(range(m), key=lambda j: (distinct_ratios[j], -avg_lens[j])))
        orig_order = tuple(range(m))

        cache_tok: Dict[Tuple[int, ...], int] = {}
        B = 25 if m <= 9 else 15

        beam = [(tuple(), 0, 0)]
        for _step in range(m):
            cand = []
            for order, mask_used, _sc in beam:
                for j in range(m):
                    bit = 1 << j
                    if mask_used & bit:
                        continue
                    o2 = order + (j,)
                    s2 = token_score(o2, cache_tok)
                    cand.append((o2, mask_used | bit, s2))
            cand.sort(key=lambda x: x[2], reverse=True)
            beam = cand[:B]

        full_orders = []
        seen_orders = set()

        def add_candidate(o: Tuple[int, ...]):
            if o not in seen_orders:
                seen_orders.add(o)
                full_orders.append(o)

        for o, _, _ in beam:
            if len(o) == m:
                add_candidate(o)
        add_candidate(heur_order)
        add_candidate(orig_order)

        if not full_orders:
            full_orders = [heur_order]

        best_tok_order = max(full_orders, key=lambda o: token_score(o, cache_tok))

        def hill_climb_token(order: Tuple[int, ...], max_passes: int = 3) -> Tuple[int, ...]:
            best = order
            best_score = token_score(best, cache_tok)
            order_list = list(best)
            for _ in range(max_passes):
                improved = False
                for i in range(m):
                    for j in range(i + 1, m):
                        order_list[i], order_list[j] = order_list[j], order_list[i]
                        o2 = tuple(order_list)
                        s2 = token_score(o2, cache_tok)
                        if s2 > best_score:
                            best_score = s2
                            best = o2
                            improved = True
                        else:
                            order_list[i], order_list[j] = order_list[j], order_list[i]
                    if improved:
                        order_list = list(best)
                if not improved:
                    break
            return best

        best_tok_order = hill_climb_token(best_tok_order)
        add_candidate(best_tok_order)

        k_char = min(n, max(1200, min(early_stop, 8000)))
        is_ascii, col_bytes, col_strs_char = self._prepare_sample_char_data(dfm, cols, k_char)

        def char_score(order: Tuple[int, ...], cache: Dict[Tuple[int, ...], int]) -> int:
            if order in cache:
                return cache[order]

            seen = set()
            total = 0
            P = self._P_CH
            mask = self._MASK64

            if is_ascii:
                bcols = [col_bytes[i] for i in order]
                for r in range(k_char):
                    h = 1469598103934665603
                    miss = False
                    hit = 0
                    for bcol in bcols:
                        bs = bcol[r]
                        for b in bs:
                            h = (h * P + (b + 1)) & mask
                            if (not miss) and (h in seen):
                                hit += 1
                            else:
                                miss = True
                            seen.add(h)
                    if r:
                        total += hit
            else:
                scols = [col_strs_char[i] for i in order]
                for r in range(k_char):
                    h = 1469598103934665603
                    miss = False
                    hit = 0
                    for scol in scols:
                        s = scol[r]
                        for ch in s:
                            h = (h * P + (ord(ch) + 1)) & mask
                            if (not miss) and (h in seen):
                                hit += 1
                            else:
                                miss = True
                            seen.add(h)
                    if r:
                        total += hit

            cache[order] = total
            return total

        cache_char: Dict[Tuple[int, ...], int] = {}
        top_by_tok = sorted(full_orders, key=lambda o: token_score(o, cache_tok), reverse=True)[: min(10, len(full_orders))]
        for o in top_by_tok:
            add_candidate(o)

        candidates = list(seen_orders)
        candidates.sort(key=lambda o: token_score(o, cache_tok), reverse=True)
        candidates = candidates[: min(12, len(candidates))]

        best_order = max(candidates, key=lambda o: char_score(o, cache_char))
        ordered_cols = [cols[i] for i in best_order]

        return dfm.astype(str)[ordered_cols]