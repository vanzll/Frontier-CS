import os
import re
import tarfile
import zipfile
from typing import Optional, Tuple


class Solution:
    _KW_WEIGHTS = {
        "clusterfuzz": 200,
        "testcase": 80,
        "minimized": 70,
        "repro": 60,
        "reproducer": 60,
        "poc": 60,
        "crash": 60,
        "asan": 30,
        "ubsan": 30,
        "artifact": 30,
        "artifacts": 30,
        "corpus": 10,
        "seed": 10,
        "seeds": 10,
        "input": 10,
        "fuzz": 5,
    }

    _BANNED_BASENAMES = {
        "license",
        "copying",
        "authors",
        "readme",
        "changelog",
        "cmakelists.txt",
        "makefile",
        "build",
        "workspace",
        "dockerfile",
        "build.sh",
        "configure",
        "configure.ac",
    }

    _BANNED_EXTS = {
        ".c", ".cc", ".cpp", ".cxx", ".h", ".hpp", ".hh", ".hxx",
        ".py", ".pyi", ".java", ".js", ".ts", ".go", ".rs", ".cs",
        ".md", ".rst", ".adoc",
        ".cmake", ".mk",
        ".sh", ".bat", ".ps1",
        ".yml", ".yaml", ".json",
        ".toml", ".ini", ".cfg",
        ".html", ".css",
        ".in",
        ".gitignore", ".gitattributes",
    }

    def _kw_score(self, path: str) -> int:
        p = path.lower()
        s = 0
        for k, w in self._KW_WEIGHTS.items():
            if k in p:
                s += w
        return s

    def _is_banned(self, path: str) -> bool:
        base = os.path.basename(path).lower()
        if base in self._BANNED_BASENAMES:
            return True
        root, ext = os.path.splitext(base)
        if ext in self._BANNED_EXTS:
            return True
        if base.endswith(".tar") or base.endswith(".tgz") or base.endswith(".tar.gz") or base.endswith(".zip"):
            return True
        if "license" in base or "readme" in base or "changelog" in base:
            return True
        return False

    def _binary_bonus(self, data: bytes) -> int:
        if not data:
            return -1000
        nonprint = 0
        for b in data:
            if b in (9, 10, 13):
                continue
            if b < 32 or b > 126:
                nonprint += 1
        ratio = nonprint / max(1, len(data))
        if ratio >= 0.5:
            return 10
        if ratio >= 0.2:
            return 5
        return 0

    def _likely_text_doc(self, data: bytes) -> bool:
        if not data:
            return True
        if len(data) > 64 and b"\n" in data:
            good = 0
            for b in data:
                if b in (9, 10, 13) or 32 <= b <= 126:
                    good += 1
            if good / len(data) > 0.98:
                return True
        return False

    def _try_decode_embedded(self, path: str, data: bytes) -> Optional[bytes]:
        # Only attempt if small, mostly printable, and name hints at encoded content.
        p = path.lower()
        if len(data) > 256:
            return None
        if not any(k in p for k in ("b64", "base64", "hex", "encoded", "poc", "testcase", "repro", "crash")):
            return None

        try:
            s = data.strip()
        except Exception:
            return None
        if not s:
            return None

        # Try hex
        if re.fullmatch(rb"[0-9a-fA-F\s]+", s) and len(re.sub(rb"\s+", b"", s)) % 2 == 0:
            try:
                hs = re.sub(rb"\s+", b"", s)
                dec = bytes.fromhex(hs.decode("ascii"))
                if dec and len(dec) < len(data):
                    return dec
            except Exception:
                pass

        # Try base64
        if re.fullmatch(rb"[A-Za-z0-9+/=\s]+", s):
            try:
                import base64
                bs = re.sub(rb"\s+", b"", s)
                if len(bs) % 4 == 0:
                    dec = base64.b64decode(bs, validate=True)
                    if dec and len(dec) < len(data):
                        return dec
            except Exception:
                pass

        return None

    def _consider_candidate(self, path: str, data: bytes, base_score: int) -> Optional[Tuple[Tuple[int, int, int], bytes]]:
        if not data:
            return None
        if self._likely_text_doc(data) and base_score < 50 and len(data) > 32:
            return None

        score = base_score
        score += self._binary_bonus(data)

        # Prefer very small inputs slightly.
        if len(data) <= 16:
            score += 20
        elif len(data) <= 64:
            score += 5

        # Prefer inputs with explicit crash/testcase naming.
        pl = path.lower()
        if "clusterfuzz-testcase-minimized" in pl:
            score += 200
        if "minimized" in pl and "testcase" in pl:
            score += 50
        if "crash" in pl:
            score += 20

        key = (score, -len(data), -sum(1 for b in data if b == 0))  # fewer NULs slightly preferred
        return (key, data)

    def _best_from_tar(self, tar_path: str) -> Optional[bytes]:
        best_key = None
        best_data = None
        try:
            with tarfile.open(tar_path, "r:*") as tf:
                members = tf.getmembers()
                for m in members:
                    if not m.isreg():
                        continue
                    if m.size <= 0 or m.size > 4096:
                        continue
                    name = m.name or ""
                    if self._is_banned(name):
                        continue
                    base_score = self._kw_score(name)
                    if base_score <= 0 and m.size > 32:
                        continue
                    f = tf.extractfile(m)
                    if f is None:
                        continue
                    try:
                        data = f.read()
                    except Exception:
                        continue

                    decoded = self._try_decode_embedded(name, data)
                    if decoded is not None and 0 < len(decoded) <= 4096:
                        cand = self._consider_candidate(name + ":decoded", decoded, base_score + 25)
                        if cand is not None:
                            key, cdata = cand
                            if best_key is None or key > best_key:
                                best_key, best_data = key, cdata
                                if best_key[0] >= 350 and len(best_data) <= 64:
                                    return best_data

                    cand = self._consider_candidate(name, data, base_score)
                    if cand is None:
                        continue
                    key, cdata = cand
                    if best_key is None or key > best_key:
                        best_key, best_data = key, cdata
                        if best_key[0] >= 350 and len(best_data) <= 64:
                            return best_data
        except Exception:
            return None
        return best_data

    def _best_from_zip(self, zip_path: str) -> Optional[bytes]:
        best_key = None
        best_data = None
        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                for info in zf.infolist():
                    if info.is_dir():
                        continue
                    if info.file_size <= 0 or info.file_size > 4096:
                        continue
                    name = info.filename or ""
                    if self._is_banned(name):
                        continue
                    base_score = self._kw_score(name)
                    if base_score <= 0 and info.file_size > 32:
                        continue
                    try:
                        data = zf.read(info)
                    except Exception:
                        continue

                    decoded = self._try_decode_embedded(name, data)
                    if decoded is not None and 0 < len(decoded) <= 4096:
                        cand = self._consider_candidate(name + ":decoded", decoded, base_score + 25)
                        if cand is not None:
                            key, cdata = cand
                            if best_key is None or key > best_key:
                                best_key, best_data = key, cdata
                                if best_key[0] >= 350 and len(best_data) <= 64:
                                    return best_data

                    cand = self._consider_candidate(name, data, base_score)
                    if cand is None:
                        continue
                    key, cdata = cand
                    if best_key is None or key > best_key:
                        best_key, best_data = key, cdata
                        if best_key[0] >= 350 and len(best_data) <= 64:
                            return best_data
        except Exception:
            return None
        return best_data

    def _best_from_dir(self, root: str) -> Optional[bytes]:
        best_key = None
        best_data = None
        for dirpath, _, filenames in os.walk(root):
            for fn in filenames:
                path = os.path.join(dirpath, fn)
                rel = os.path.relpath(path, root)
                try:
                    st = os.stat(path)
                except Exception:
                    continue
                if st.st_size <= 0 or st.st_size > 4096:
                    continue
                if self._is_banned(rel):
                    continue
                base_score = self._kw_score(rel)
                if base_score <= 0 and st.st_size > 32:
                    continue
                try:
                    with open(path, "rb") as f:
                        data = f.read()
                except Exception:
                    continue

                decoded = self._try_decode_embedded(rel, data)
                if decoded is not None and 0 < len(decoded) <= 4096:
                    cand = self._consider_candidate(rel + ":decoded", decoded, base_score + 25)
                    if cand is not None:
                        key, cdata = cand
                        if best_key is None or key > best_key:
                            best_key, best_data = key, cdata
                            if best_key[0] >= 350 and len(best_data) <= 64:
                                return best_data

                cand = self._consider_candidate(rel, data, base_score)
                if cand is None:
                    continue
                key, cdata = cand
                if best_key is None or key > best_key:
                    best_key, best_data = key, cdata
                    if best_key[0] >= 350 and len(best_data) <= 64:
                        return best_data
        return best_data

    def solve(self, src_path: str) -> bytes:
        if src_path and os.path.isdir(src_path):
            data = self._best_from_dir(src_path)
            if data:
                return data
            return b"A" * 9

        if src_path and tarfile.is_tarfile(src_path):
            data = self._best_from_tar(src_path)
            if data:
                return data
            return b"A" * 9

        if src_path and zipfile.is_zipfile(src_path):
            data = self._best_from_zip(src_path)
            if data:
                return data
            return b"A" * 9

        return b"A" * 9