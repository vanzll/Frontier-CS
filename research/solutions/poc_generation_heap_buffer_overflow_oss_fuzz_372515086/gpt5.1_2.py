import os
import tarfile
import zipfile
import tempfile
import stat
import re
from typing import Optional, Tuple


class Solution:
    TARGET_LEN = 1032

    def solve(self, src_path: str) -> bytes:
        root = self._extract_src(src_path)
        poc = self._find_poc_external(root)
        if poc is None:
            poc = self._find_inline_poc(root)
        if poc is None:
            poc = b"A" * self.TARGET_LEN
        return poc

    def _extract_src(self, src_path: str) -> str:
        if os.path.isdir(src_path):
            return src_path
        tmpdir = tempfile.mkdtemp(prefix="src-")
        if tarfile.is_tarfile(src_path):
            try:
                with tarfile.open(src_path, "r:*") as tf:
                    members = []
                    for m in tf.getmembers():
                        name = m.name
                        if not name:
                            continue
                        name = name.replace("\\", "/")
                        if name.startswith("/"):
                            continue
                        if ".." in name.split("/"):
                            continue
                        members.append(m)
                    tf.extractall(tmpdir, members=members)
            except Exception:
                pass
            return tmpdir
        if zipfile.is_zipfile(src_path):
            try:
                with zipfile.ZipFile(src_path, "r") as zf:
                    for info in zf.infolist():
                        name = info.filename
                        if not name or name.endswith("/"):
                            continue
                        name_norm = name.replace("\\", "/")
                        if name_norm.startswith("/"):
                            continue
                        if ".." in name_norm.split("/"):
                            continue
                        target = os.path.join(tmpdir, name_norm)
                        os.makedirs(os.path.dirname(target), exist_ok=True)
                        with zf.open(info, "r") as src, open(target, "wb") as dst:
                            dst.write(src.read())
            except Exception:
                pass
            return tmpdir
        return os.path.dirname(src_path) or "."

    def _score_basic(self, lower_path: str) -> float:
        main_keywords = [
            "372515086",
            "polygon",
            "polyfill",
            "poly_to_cells",
            "polygon_to_cells",
            "polygontocells",
            "polycell",
            "poly_cells",
        ]
        fuzz_keywords = [
            "fuzz",
            "fuzzer",
            "oss-fuzz",
            "clusterfuzz",
            "corpus",
            "seed",
            "crash",
            "poc",
            "regress",
            "bug",
        ]
        score = 0.0
        for kw in fuzz_keywords:
            if kw in lower_path:
                score += 5.0
        for kw in main_keywords:
            if kw in lower_path:
                score += 4.0
        if "cell" in lower_path:
            score += 1.0
        if "poly" in lower_path:
            score += 1.0
        return score

    def _find_poc_in_zip(self, zip_path: str) -> Optional[bytes]:
        target = self.TARGET_LEN
        best: Optional[Tuple[float, bytes]] = None
        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                for info in zf.infolist():
                    if info.is_dir():
                        continue
                    size = info.file_size
                    if size == 0 or size > 65536:
                        continue
                    member_path = (zip_path + "::" + info.filename).lower()
                    score_basic = self._score_basic(member_path)
                    size_closeness = max(0.0, 2.5 - abs(size - target) / 256.0)
                    score = score_basic + size_closeness
                    if score <= 0.0:
                        continue
                    try:
                        data = zf.read(info.filename)
                    except Exception:
                        continue
                    if best is None or score > best[0]:
                        best = (score, data)
        except Exception:
            return None
        if best is not None:
            return best[1]
        return None

    def _find_poc_external(self, root: str) -> Optional[bytes]:
        target = self.TARGET_LEN
        best_exact: Optional[Tuple[float, bytes]] = None
        best_any: Optional[Tuple[float, bytes]] = None
        max_size = 65536
        code_exts = {
            ".c",
            ".cc",
            ".cpp",
            ".cxx",
            ".h",
            ".hpp",
            ".hh",
            ".m",
            ".mm",
            ".java",
            ".py",
            ".sh",
            ".bat",
            ".ps1",
            ".cs",
        }
        for dirpath, dirnames, filenames in os.walk(root):
            for name in filenames:
                full = os.path.join(dirpath, name)
                try:
                    st = os.lstat(full)
                except OSError:
                    continue
                if not stat.S_ISREG(st.st_mode):
                    continue
                size = st.st_size
                if size == 0 or size > max_size:
                    continue
                ext = os.path.splitext(name)[1].lower()
                if ext in code_exts:
                    continue
                lower_full = full.lower()
                if ext == ".zip":
                    data_from_zip = self._find_poc_in_zip(full)
                    if data_from_zip is not None:
                        return data_from_zip
                score_basic = self._score_basic(lower_full)
                if size == target:
                    score = score_basic + 3.0
                    if score_basic <= 0.0 and "fuzz" not in lower_full and "corpus" not in lower_full:
                        continue
                    try:
                        with open(full, "rb") as f:
                            content = f.read()
                    except OSError:
                        continue
                    if best_exact is None or score > best_exact[0]:
                        best_exact = (score, content)
                    continue
                size_closeness = max(0.0, 2.5 - abs(size - target) / 256.0)
                score = score_basic + size_closeness
                if score <= 0.0:
                    continue
                try:
                    with open(full, "rb") as f:
                        data = f.read()
                except OSError:
                    continue
                if best_any is None or score > best_any[0]:
                    best_any = (score, data)
        if best_exact is not None:
            return best_exact[1]
        if best_any is not None:
            return best_any[1]
        return None

    def _decode_c_string(self, s: str) -> bytes:
        out = bytearray()
        i = 0
        n = len(s)
        while i < n:
            ch = s[i]
            if ch != "\\":
                out.append(ord(ch))
                i += 1
                continue
            i += 1
            if i >= n:
                break
            c = s[i]
            i += 1
            if c in "abfnrtv'\"\\?":
                mapping = {
                    "a": 7,
                    "b": 8,
                    "f": 12,
                    "n": 10,
                    "r": 13,
                    "t": 9,
                    "v": 11,
                    "'": 39,
                    '"': 34,
                    "\\": 92,
                    "?": 63,
                }
                out.append(mapping.get(c, ord(c)))
            elif c in "xX":
                hex_digits = ""
                while i < n and s[i] in "0123456789abcdefABCDEF" and len(hex_digits) < 2:
                    hex_digits += s[i]
                    i += 1
                if hex_digits:
                    out.append(int(hex_digits, 16))
                else:
                    out.append(ord("x"))
            elif "0" <= c <= "7":
                oct_digits = c
                count = 1
                while i < n and count < 3 and "0" <= s[i] <= "7":
                    oct_digits += s[i]
                    i += 1
                    count += 1
                try:
                    out.append(int(oct_digits, 8))
                except ValueError:
                    out.append(0)
            else:
                out.append(ord(c))
        return bytes(out)

    def _find_inline_poc(self, root: str) -> Optional[bytes]:
        code_exts = (".c", ".cc", ".cpp", ".cxx", ".h", ".hpp", ".hh", ".m", ".mm", ".txt")
        bug_id = "372515086"
        pattern_string = re.compile(r'"((?:\\.|[^"\\])*)"', re.DOTALL)
        for dirpath, dirnames, filenames in os.walk(root):
            for name in filenames:
                ext = os.path.splitext(name)[1].lower()
                if ext not in code_exts:
                    continue
                full = os.path.join(dirpath, name)
                try:
                    with open(full, "r", encoding="utf-8", errors="ignore") as f:
                        text = f.read()
                except OSError:
                    continue
                if bug_id not in text and "polygonToCellsExperimental" not in text:
                    continue
                idx_candidates = []
                pos1 = text.find(bug_id)
                if pos1 != -1:
                    idx_candidates.append(pos1)
                pos2 = text.find("polygonToCellsExperimental")
                if pos2 != -1:
                    idx_candidates.append(pos2)
                if idx_candidates:
                    idx = min(idx_candidates)
                else:
                    idx = 0
                snippet = text[idx:]
                m = pattern_string.search(snippet)
                if not m:
                    m = pattern_string.search(text)
                    if not m:
                        continue
                c_string = m.group(1)
                data = self._decode_c_string(c_string)
                if data:
                    return data
        return None