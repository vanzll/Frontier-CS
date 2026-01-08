import struct
import zlib

class Solution:
    def solve(self, src_path: str) -> bytes:
        return self._png_zero_width()

    def _png_chunk(self, chunk_type: bytes, data: bytes) -> bytes:
        length = struct.pack(">I", len(data))
        crc = struct.pack(">I", zlib.crc32(chunk_type + data) & 0xffffffff)
        return length + chunk_type + data + crc

    def _png_zero_width(self) -> bytes:
        # PNG signature
        sig = b'\x89PNG\r\n\x1a\n'
        # IHDR: width=0, height=2, bit depth=8, color type=2 (truecolor), compression=0, filter=0, interlace=0
        ihdr_data = struct.pack(">IIBBBBB", 0, 2, 8, 2, 0, 0, 0)
        ihdr = self._png_chunk(b'IHDR', ihdr_data)
        # IDAT: two filter bytes for two rows (each 0 = "None" filter), zero pixels since width=0
        idat_data_uncompressed = b'\x00\x00'
        idat_compressed = zlib.compress(idat_data_uncompressed, level=9)
        idat = self._png_chunk(b'IDAT', idat_compressed)
        # IEND
        iend = self._png_chunk(b'IEND', b'')
        return sig + ihdr + idat + iend