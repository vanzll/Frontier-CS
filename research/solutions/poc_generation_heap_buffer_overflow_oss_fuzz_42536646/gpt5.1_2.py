import struct
import zlib


class Solution:
    def solve(self, src_path: str) -> bytes:
        def build_chunk(chunk_type: bytes, data: bytes) -> bytes:
            length = struct.pack(">I", len(data))
            crc = zlib.crc32(chunk_type + data) & 0xFFFFFFFF
            crc_bytes = struct.pack(">I", crc)
            return length + chunk_type + data + crc_bytes

        # Zero-width PNG with non-zero height
        width = 0
        height = 64  # any positive value

        signature = b"\x89PNG\r\n\x1a\n"

        # IHDR: width, height, bit depth 8, color type 6 (RGBA), compression 0, filter 0, interlace 0
        ihdr_data = struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)
        ihdr_chunk = build_chunk(b"IHDR", ihdr_data)

        # For width=0 and height>0, each row is just a filter byte (0x00)
        raw_data = b"\x00" * height  # height scanlines, 1 filter byte each
        compressed_data = zlib.compress(raw_data)
        idat_chunk = build_chunk(b"IDAT", compressed_data)

        iend_chunk = build_chunk(b"IEND", b"")

        return signature + ihdr_chunk + idat_chunk + iend_chunk