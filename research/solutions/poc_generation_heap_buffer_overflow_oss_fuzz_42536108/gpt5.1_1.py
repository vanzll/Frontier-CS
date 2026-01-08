import struct


class Solution:
    def solve(self, src_path: str) -> bytes:
        prefix = b'A' * 24
        eocd = (
            b'PK\x05\x06' +
            struct.pack(
                '<HHHHIIH',
                0,   # number of this disk
                0,   # disk where central directory starts
                1,   # number of central directory records on this disk
                1,   # total number of central directory records
                16,  # size of central directory (bytes)
                32,  # offset of start of central directory, relative to start of archive
                0    # comment length
            )
        )
        return prefix + eocd