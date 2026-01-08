import os
import re
import tarfile
from typing import Optional, Tuple


class Solution:
    def _read_text_member(self, tf: tarfile.TarFile, member: tarfile.TarInfo, max_bytes: int = 2_000_000) -> Optional[str]:
        try:
            f = tf.extractfile(member)
            if f is None:
                return None
            data = f.read(max_bytes)
            if not data:
                return ""
            return data.decode("utf-8", errors="ignore")
        except Exception:
            return None

    def _extract_min_size_and_prefix(self, text: str) -> Tuple[int, Optional[bytes]]:
        min_size = 0
        prefix = None

        # Find minimal size checks (common in fuzzers)
        for m in re.finditer(r'if\s*\(\s*(?:size|Size|len|length)\s*<\s*(\d+)\s*\)\s*return', text):
            try:
                n = int(m.group(1))
                if n > min_size:
                    min_size = n
            except Exception:
                pass

        # Find explicit prefix requirements via memcmp/strncmp against data
        # Heuristic: look for `if (memcmp(data, "XYZ", N) != 0) return;`
        pat = re.compile(
            r'if\s*\(\s*(?:memcmp|strncmp)\s*\(\s*(?:\(const\s+char\s*\*\)\s*)?data\s*,\s*"([^"]+)"\s*,\s*(\d+)\s*\)\s*!=\s*0\s*\)\s*return',
            re.MULTILINE,
        )
        for m in pat.finditer(text):
            s = m.group(1)
            try:
                n = int(m.group(2))
            except Exception:
                continue
            b = s.encode("utf-8", errors="ignore")
            if n <= len(b):
                b = b[:n]
            else:
                # If asked length exceeds literal, skip
                continue
            if prefix is None or len(b) > len(prefix):
                prefix = b

        # Another pattern: `if (memcmp(&data[0], "XYZ", 3)) return;`
        pat2 = re.compile(
            r'if\s*\(\s*(?:memcmp|strncmp)\s*\(\s*&\s*data\s*\[\s*0\s*\]\s*,\s*"([^"]+)"\s*,\s*(\d+)\s*\)\s*!=\s*0\s*\)\s*return',
            re.MULTILINE,
        )
        for m in pat2.finditer(text):
            s = m.group(1)
            try:
                n = int(m.group(2))
            except Exception:
                continue
            b = s.encode("utf-8", errors="ignore")
            if n <= len(b):
                b = b[:n]
            else:
                continue
            if prefix is None or len(b) > len(prefix):
                prefix = b

        return min_size, prefix

    def solve(self, src_path: str) -> bytes:
        target_len = 9
        min_size = 0
        best_prefix = None

        if os.path.isfile(src_path):
            try:
                with tarfile.open(src_path, "r:*") as tf:
                    for member in tf.getmembers():
                        if not member.isfile():
                            continue
                        name = member.name.lower()
                        if not (name.endswith((".c", ".cc", ".cpp", ".cxx", ".h", ".hpp")) or "fuzz" in name or "fuzzer" in name):
                            continue
                        text = self._read_text_member(tf, member)
                        if text is None:
                            continue
                        ms, pref = self._extract_min_size_and_prefix(text)
                        if ms > min_size:
                            min_size = ms
                        if pref is not None:
                            if best_prefix is None or len(pref) > len(best_prefix):
                                best_prefix = pref
            except Exception:
                pass

        length = max(target_len, min_size if min_size > 0 else target_len)

        if best_prefix is not None and len(best_prefix) <= length:
            payload = best_prefix + (b"A" * (length - len(best_prefix)))
        else:
            payload = b"A" * length

        # Prefer exactly 9 bytes when possible
        if length > target_len and min_size <= target_len:
            payload = payload[:target_len]

        return payload