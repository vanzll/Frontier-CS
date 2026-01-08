import struct
import binascii
import zlib
import tarfile
from typing import Optional


class Solution:
    def solve(self, src_path: str) -> bytes:
        fmt = self._detect_format(src_path)
        if fmt == "gif":
            return self._make_gif_zero_width()
        # Default to PNG
        return self._make_png_zero_width()

    def _detect_format(self, src_path: str) -> str:
        """
        Very lightweight heuristic to detect the image format/library from the
        source tarball. Currently distinguishes GIF vs "everything else",
        where everything else is treated as PNG.
        """
        try:
            tf = tarfile.open(src_path, "r:*")
        except Exception:
            return "png"

        fmt: Optional[str] = None
        try:
            for member in tf.getmembers():
                name = member.name.lower()

                # Prefer strong GIF indicators
                if any(x in name for x in ("giflib", "gif_lib.h", "dgif_lib.c", "egif_lib.c")):
                    return "gif"

                # PNG / libspng / libpng indicators
                if any(x in name for x in ("libspng", "spng.h", "png.h", "libpng")):
                    if fmt is None:
                        fmt = "png"

                # WebP indicators (not used yet, but kept for extensibility)
                if any(x in name for x in ("libwebp", "webp/decode.h", "webp/encode.h")):
                    if fmt is None:
                        fmt = "png"  # treat as PNG-like for our purposes

            return fmt or "png"
        finally:
            try:
                tf.close()
            except Exception:
                pass

    def _make_png_zero_width(self) -> bytes:
        """
        Construct a minimal PNG image whose IHDR declares width=0 and height=1.
        This is invalid per the PNG spec but often exercises dimension-handling
        code paths.
        """
        # PNG signature
        signature = b"\x89PNG\r\n\x1a\n"

        # IHDR with width=0, height=1, 8-bit RGBA
        width = 0
        height = 1
        bit_depth = 8
        color_type = 6  # RGBA
        compression_method = 0
        filter_method = 0
        interlace_method = 0

        ihdr_data = struct.pack(
            ">IIBBBBB",
            width,
            height,
            bit_depth,
            color_type,
            compression_method,
            filter_method,
            interlace_method,
        )
        ihdr_type = b"IHDR"
        ihdr_len = struct.pack(">I", len(ihdr_data))
        ihdr_crc = struct.pack(">I", binascii.crc32(ihdr_type + ihdr_data) & 0xFFFFFFFF)
        ihdr_chunk = ihdr_len + ihdr_type + ihdr_data + ihdr_crc

        # One scanline: filter byte (0) + 4 bytes RGBA pixel
        raw_scanline = b"\x00\x00\x00\x00\x00"  # filter=0, pixel=(0,0,0,0)
        compressed_scanline = zlib.compress(raw_scanline)

        idat_type = b"IDAT"
        idat_len = struct.pack(">I", len(compressed_scanline))
        idat_crc = struct.pack(">I", binascii.crc32(idat_type + compressed_scanline) & 0xFFFFFFFF)
        idat_chunk = idat_len + idat_type + compressed_scanline + idat_crc

        # IEND chunk
        iend_type = b"IEND"
        iend_len = struct.pack(">I", 0)
        iend_crc = struct.pack(">I", binascii.crc32(iend_type) & 0xFFFFFFFF)
        iend_chunk = iend_len + iend_type + iend_crc

        return signature + ihdr_chunk + idat_chunk + iend_chunk

    def _make_gif_zero_width(self) -> bytes:
        """
        Construct a minimal GIF89a image where both the logical screen width
        and the image descriptor width are zero.
        Based on a standard 1x1 GIF with dimensions forced to zero.
        """
        # This is derived from a canonical minimal GIF:
        # GIF89a, logical screen 1x1, global color table with 2 colors,
        # a single image descriptor 1x1, and minimal LZW data.
        # We modify widths to 0 while keeping everything else intact.
        data = bytearray([
            # Header: GIF89a
            0x47, 0x49, 0x46, 0x38, 0x39, 0x61,
            # Logical Screen Descriptor
            # Width  = 0 (LSB, MSB)
            0x00, 0x00,
            # Height = 1
            0x01, 0x00,
            # GCT flag set, color resolution, sort flag, GCT size (2 colors)
            0x80,
            # Background color index
            0x00,
            # Pixel aspect ratio
            0x00,
            # Global Color Table (2 entries, 3 bytes each)
            0x00, 0x00, 0x00,        # Color #0: black
            0xFF, 0xFF, 0xFF,        # Color #1: white

            # Image Descriptor
            0x2C,                    # Image Separator
            0x00, 0x00,              # Image Left Position
            0x00, 0x00,              # Image Top Position
            # Image Width  = 0
            0x00, 0x00,
            # Image Height = 1
            0x01, 0x00,
            # No local color table, non-interlaced
            0x00,

            # Image Data
            0x02,                    # LZW minimum code size
            0x02,                    # Block size
            0x4C, 0x01,              # Image data
            0x00,                    # Block terminator

            # Trailer
            0x3B,
        ])
        return bytes(data)