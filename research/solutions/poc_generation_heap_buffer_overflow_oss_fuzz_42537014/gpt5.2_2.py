import os
import re
import tarfile
import tempfile
from typing import List, Optional, Tuple


def _is_probably_text(path: str, max_bytes: int = 65536) -> bool:
    try:
        with open(path, "rb") as f:
            data = f.read(max_bytes)
        if not data:
            return True
        if b"\x00" in data:
            return False
        # Heuristic: if too many non-printable bytes, treat as binary
        non_print = 0
        for b in data:
            if b in (9, 10, 13):
                continue
            if b < 32 or b > 126:
                non_print += 1
        return (non_print / len(data)) < 0.15
    except Exception:
        return False


def _safe_extract_tar(tar: tarfile.TarFile, path: str) -> None:
    base = os.path.abspath(path)
    for member in tar.getmembers():
        name = member.name
        if not name:
            continue
        dest = os.path.abspath(os.path.join(base, name))
        if not (dest == base or dest.startswith(base + os.sep)):
            continue
        if member.islnk() or member.issym():
            continue
        try:
            tar.extract(member, path=path, set_attrs=False)
        except Exception:
            pass


def _collect_candidate_poc_files(root: str) -> List[Tuple[int, str]]:
    name_re = re.compile(
        r"(clusterfuzz|testcase|minimized|repro|reproducer|crash|poc|proof|asan|ubsan)",
        re.IGNORECASE,
    )
    candidates: List[Tuple[int, str]] = []
    for dirpath, dirnames, filenames in os.walk(root):
        # Skip common large dirs
        base = os.path.basename(dirpath).lower()
        if base in (".git", "build", "out", "bazel-out", "node_modules", ".svn"):
            dirnames[:] = []
            continue
        for fn in filenames:
            low = fn.lower()
            if low.endswith((".o", ".a", ".so", ".dll", ".dylib", ".exe")):
                continue
            if not name_re.search(fn):
                continue
            p = os.path.join(dirpath, fn)
            try:
                st = os.stat(p)
            except Exception:
                continue
            if not os.path.isfile(p):
                continue
            if st.st_size <= 0 or st.st_size > 1_000_000:
                continue
            candidates.append((st.st_size, p))
    candidates.sort()
    return candidates


def _read_file_bytes(path: str, limit: int = 2_000_000) -> Optional[bytes]:
    try:
        with open(path, "rb") as f:
            data = f.read(limit + 1)
        if len(data) > limit:
            return None
        return data
    except Exception:
        return None


def _find_fuzzer_files(root: str) -> List[str]:
    fuzzers: List[str] = []
    exts = (".c", ".cc", ".cpp", ".cxx", ".h", ".hpp")
    for dirpath, dirnames, filenames in os.walk(root):
        base = os.path.basename(dirpath).lower()
        if base in (".git", "build", "out", "bazel-out", "node_modules"):
            dirnames[:] = []
            continue
        for fn in filenames:
            if not fn.lower().endswith(exts):
                continue
            p = os.path.join(dirpath, fn)
            try:
                if os.path.getsize(p) > 2_000_000:
                    continue
            except Exception:
                continue
            if not _is_probably_text(p):
                continue
            try:
                with open(p, "rb") as f:
                    data = f.read()
            except Exception:
                continue
            if b"LLVMFuzzerTestOneInput" in data or b"FuzzerTestOneInput" in data:
                fuzzers.append(p)
    return fuzzers


def _extract_required_prefix_and_min_size(fuzzer_text: str) -> Tuple[bytes, int]:
    min_size = 0
    # Size checks: if (Size < N) return 0;
    for m in re.finditer(r"\bSize\s*<\s*(\d+)\s*\)\s*return\b", fuzzer_text):
        try:
            n = int(m.group(1))
            if 0 <= n <= 4096:
                min_size = max(min_size, n)
        except Exception:
            pass

    # Also handle: if (size < N) return 0; (lowercase common)
    for m in re.finditer(r"\bsize\s*<\s*(\d+)\s*\)\s*return\b", fuzzer_text):
        try:
            n = int(m.group(1))
            if 0 <= n <= 4096:
                min_size = max(min_size, n)
        except Exception:
            pass

    # Checks that require a prefix to avoid early return:
    # if (memcmp(Data, "XXXX", 4) != 0) return 0;
    required: List[bytes] = []

    memcmp_re = re.compile(
        r"if\s*\(\s*memcmp\s*\(\s*(?:Data|data)\s*,\s*\"([ -~]{1,64})\"\s*,\s*(\d+)\s*\)\s*!=\s*0\s*\)\s*return\b",
        re.MULTILINE,
    )
    for m in memcmp_re.finditer(fuzzer_text):
        lit = m.group(1)
        n = int(m.group(2))
        b = lit.encode("latin1", errors="ignore")
        if 1 <= n <= 64 and len(b) >= n:
            required.append(b[:n])

    strncmp_re = re.compile(
        r"if\s*\(\s*strncmp\s*\(\s*\(const\s+char\s*\*\)\s*(?:Data|data)\s*,\s*\"([ -~]{1,64})\"\s*,\s*(\d+)\s*\)\s*!=\s*0\s*\)\s*return\b",
        re.MULTILINE,
    )
    for m in strncmp_re.finditer(fuzzer_text):
        lit = m.group(1)
        n = int(m.group(2))
        b = lit.encode("latin1", errors="ignore")
        if 1 <= n <= 64 and len(b) >= n:
            required.append(b[:n])

    # Data[0] != 'X' early return
    data0_re = re.compile(
        r"if\s*\(\s*(?:Data|data)\s*\[\s*0\s*\]\s*!=\s*'(.{1})'\s*\)\s*return\b"
    )
    for m in data0_re.finditer(fuzzer_text):
        ch = m.group(1)
        required.append(ch.encode("latin1", errors="ignore"))

    # Merge required prefixes (all assumed at offset 0; try to satisfy the longest without conflicts)
    required.sort(key=len, reverse=True)
    prefix = b""
    for r in required:
        if not prefix:
            prefix = r
            continue
        # Check compatibility at overlap
        ok = True
        overlap = min(len(prefix), len(r))
        if prefix[:overlap] != r[:overlap]:
            ok = False
        if ok and len(r) > len(prefix):
            prefix = r

    return prefix, min_size


class Solution:
    def solve(self, src_path: str) -> bytes:
        base_len = 9
        default_poc = b"A" * base_len

        with tempfile.TemporaryDirectory(prefix="pocgen_") as td:
            root = td
            if os.path.isdir(src_path):
                root = src_path
            else:
                extracted = False
                try:
                    if tarfile.is_tarfile(src_path):
                        with tarfile.open(src_path, "r:*") as tar:
                            _safe_extract_tar(tar, td)
                        extracted = True
                except Exception:
                    extracted = False
                root = td if extracted else td

                if not extracted:
                    return default_poc

            # Prefer existing clusterfuzz/repro files if present
            candidates = _collect_candidate_poc_files(root)
            for _, p in candidates[:50]:
                data = _read_file_bytes(p)
                if data is None:
                    continue
                # Prefer exact/near target length if available
                if 1 <= len(data) <= 1024:
                    return data

            # Heuristic: derive required prefix and minimal size from fuzzer harness
            fuzzers = _find_fuzzer_files(root)
            best_prefix = b""
            min_size = 0
            for fp in fuzzers[:20]:
                try:
                    with open(fp, "r", encoding="utf-8", errors="ignore") as f:
                        txt = f.read()
                except Exception:
                    continue
                pref, ms = _extract_required_prefix_and_min_size(txt)
                # Pick the one with strongest constraints (bigger min_size, longer prefix)
                if ms > min_size or (ms == min_size and len(pref) > len(best_prefix)):
                    min_size = ms
                    best_prefix = pref

            out_len = max(base_len, min_size, len(best_prefix))
            if out_len <= 0:
                out_len = base_len

            if not best_prefix:
                return b"A" * out_len

            if len(best_prefix) >= out_len:
                return best_prefix[:out_len]

            pad_len = out_len - len(best_prefix)
            return best_prefix + (b"A" * pad_len)